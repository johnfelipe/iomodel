# Copyright 2018 Twin Tech Labs. All rights reserved

from flask import Blueprint, redirect, render_template, current_app
from flask import request, url_for, flash, send_from_directory, jsonify, render_template_string
from flask_user import current_user, login_required, roles_accepted
from werkzeug.utils import secure_filename
import turicreate as tc
from turicreate import SArray
import sys
import numpy as np
from scipy import stats as scipy_stats
import psycopg2

from app import db
from app.models.user_models import UserProfileForm, UserDataForm, UserData, TrainModelForm, TrainedModel, User, ErrorLog, UsersRoles, Project
import uuid, json, os
import datetime

# When using a Flask app factory we must use a blueprint to avoid needing 'app' for '@app.route'
main_blueprint = Blueprint('main', __name__, template_folder='templates')

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def nan_to_null(f,
        _NULL=psycopg2.extensions.AsIs('NULL'),
        _NaN=np.NaN,
        _Float=psycopg2.extensions.Float):
    print(f)
    if not np.isnan(f):
        return f
    return None

psycopg2.extensions.register_adapter(float, nan_to_null)

# The User page is accessible to authenticated users (users that have logged in)
@main_blueprint.route('/member')
def member_page():
    if not current_user.is_authenticated:
        return redirect(url_for('user.login'))
    return render_template('pages/member_base.html')

# The Admin page is accessible to users with the 'admin' role
@main_blueprint.route('/admin')
@roles_accepted('admin')  # Limits access to users with the 'admin' role
def admin_page():
    return render_template('pages/admin_page.html')

@main_blueprint.route('/users')
@roles_accepted('admin')
def user_admin_page():
    users = User.query.all()
    return render_template('pages/admin/users.html',
        users=users)

@main_blueprint.route('/create_user', methods=['GET', 'POST'])
@roles_accepted('admin')
def create_user_page():
    form = UserProfileForm(request.form, obj=current_user)

    if request.method == 'POST':
        user = User.query.filter(User.email == request.form['email']).first()
        if not user:
            user = User(email=request.form['email'],
                        first_name=request.form['first_name'],
                        last_name=request.form['last_name'],
                        password=current_app.user_manager.hash_password(request.form['password']),
                        active=True,
                        confirmed_at=datetime.datetime.utcnow())
            db.session.add(user)
            db.session.commit()
        return redirect(url_for('main.user_admin_page'))
    return render_template('pages/admin/create_user.html',
                           form=form)

@main_blueprint.route('/delete_user', methods=['GET'])
@roles_accepted('admin')
def delete_user_page():
    try:
        user_id = request.args.get('user_id')

        db.session.query(UsersRoles).filter_by(user_id = user_id).delete()
        db.session.query(User).filter_by(id = user_id).delete()
        db.session.commit()

        flash('You successfully deleted your user!', 'success')
        return redirect(url_for('main.user_admin_page'))
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        return redirect(request.referrer)

@main_blueprint.route('/pages/profile', methods=['GET', 'POST'])
@login_required
def user_profile_page():
    # Initialize form
    form = UserProfileForm(request.form, obj=current_user)

    # Process valid POST
    if request.method == 'POST' and form.validate():
        # Copy form fields to user_profile fields
        form.populate_obj(current_user)

        # Save user_profile
        db.session.commit()

        # Redirect to home page
        return redirect(url_for('main.user_profile_page'))

    # Process GET or invalid POST
    return render_template('pages/user_profile_page.html',
                           current_user=current_user,
                           form=form)

@main_blueprint.route('/support')
@login_required
def support_page():
    return render_template('pages/support.html')

@main_blueprint.route('/howtos')
@login_required
def howto_page():
    return render_template('pages/howto.html')

@main_blueprint.route('/clear_error', methods=['GET'])
@roles_accepted('admin')
def clear_error_page():
    try:
        db.session.query(ErrorLog).delete()
        db.session.commit()

        flash('You successfully cleared your error log!', 'success')
        return redirect(url_for('main.error_log_page'))
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@main_blueprint.route('/machine_learning')
@login_required
def ml_page():
    return render_template('pages/ml.html')

@main_blueprint.route('/faq')
@login_required
def faq_page():
    return render_template('pages/faq.html')
    
@main_blueprint.route('/error_trends')
@roles_accepted('admin')
def error_trends_page():
    errors_by_day = []
    result = db.engine.execute("SELECT count(a.id) FROM (SELECT to_char(date_trunc('day', (current_date - offs)), 'YYYY-MM-DD') AS date FROM generate_series(0, 31, 1) AS offs) d LEFT OUTER JOIN log a ON d.date = to_char(date_trunc('day', a.created_at), 'YYYY-MM-DD') GROUP BY d.date ORDER BY d.date")
    for item in result:
        errors_by_day.append(int(item["count"]))
    return render_template('pages/admin/error_trends.html',
        errors_by_day=errors_by_day)

@main_blueprint.route('/error_log')
@roles_accepted('admin')
def error_log_page():
    errors = ErrorLog.query.all()
    return render_template('pages/admin/error_log.html',
        errors=errors)

@main_blueprint.route('/')
def project_page():
    if current_user is None or not current_user.is_authenticated:
        return redirect(url_for('user.login'))    
    my_projects = Project.query.filter_by(user_id=current_user.id).order_by(Project.name).all()
    return render_template('pages/projects/project_page.html',
        my_projects=my_projects)

@main_blueprint.route('/delete_project', methods=['GET'])
@login_required  # Limits access to authenticated users
def delete_project_page():
    try:
        project_id = request.args.get('project_id')

        model_count = TrainedModel.query.filter_by(project_id=project_id).count()
        data_count = UserData.query.filter_by(project_id=project_id).count()
        if model_count > 0 or data_count > 0:
            flash('Opps! It appears that you have models or data still attached to this project.  Delete them first.', 'error')
            return redirect(request.referrer)
        db.session.query(Project).filter_by(id = project_id).delete()
        db.session.commit()

        flash('You successfully deleted your project!', 'success')
        return redirect(url_for('main.project_page'))
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@main_blueprint.route('/new_project', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def new_project_page():
    try:
        project = Project()
        form = UserDataForm(request.form, obj=project)

        if request.method == 'POST':
            project.user_id = current_user.id
            project.name = target = request.form['name']
            project.description = target = request.form['description']

            db.session.add(project)
            db.session.commit()
            flash('New research project created!', 'success')
            return redirect(url_for('main.project_page', project_id=project.id))
        return render_template('pages/projects/new.html',
                               form=form)
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@main_blueprint.route('/project')
@login_required
def my_project_page():
    project_id = request.args.get('project_id')
    my_project = Project.query.filter_by(id=project_id).first()
    my_data = UserData.query.filter_by(project_id=project_id).order_by(UserData.name).all()
    my_models = TrainedModel.query.filter_by(project_id=project_id).order_by(TrainedModel.name).all()

    return render_template('pages/projects/base.html',
        my_data=my_data,
        my_project=my_project,
        my_models=my_models)