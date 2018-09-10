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
from app.models.user_models import UserProfileForm, UserDataForm, User, UsersRoles, DBConn
import uuid, json, os
import datetime
from sqlalchemy import create_engine

# When using a Flask app factory we must use a blueprint to avoid needing 'app' for '@app.route'
conn_blueprint = Blueprint('connections', __name__, template_folder='templates')

@conn_blueprint.route('/connections')
def connections_page():
    print(current_user)
    conns = DBConn.query.filter_by(user_id = current_user.id).all()
    status = {}
    for conn in conns:
        try:
            engine = create_engine(conn.engine_type + '://' + conn.user + ':' + conn.password + '@'+ conn.host + '/' + conn.db)        
            connection = engine.connect()
            status[conn.id] = "images/green.png"
        except Exception as e:
            status[conn.id] = "images/red.png"
            continue

    return render_template('pages/connections/list.html',
        status=status,
        conns=conns)

@conn_blueprint.route('/create', methods=['GET', 'POST'])
def create_page():
    form = UserProfileForm(request.form, obj=None)

    if request.method == 'POST':
        conn = DBConn(user_id=current_user.id,
                    name=request.form['name'],
                    db=request.form['db'],
                    engine_type=request.form['engine_type'],
                    host=request.form['host'],
                    user=request.form['user'],
                    password=request.form['password'])
        db.session.add(conn)
        db.session.commit()
        return redirect(url_for('connections.connections_page'))
    return render_template('pages/connections/create.html',
                           form=form)

@conn_blueprint.route('/delete', methods=['GET'])
@roles_accepted('admin')
def delete():
    try:
        conn_id = request.args.get('conn_id')

        db.session.query(DBConn).filter_by(id = conn_id).delete()
        db.session.commit()

        flash('You successfully deleted your connection!', 'success')
        return redirect(url_for('connections.connections_page'))
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        return redirect(request.referrer)