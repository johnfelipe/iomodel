#
# Copyright 2018 Twin Tech Labs
# Author: Matt Hogan
#
from flask import Blueprint, redirect, render_template, current_app
from flask import request, url_for, flash, send_from_directory, jsonify, render_template_string
from flask_user import current_user, login_required, roles_accepted
from werkzeug.utils import secure_filename
import turicreate as tc
from turicreate import SArray
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import sys
import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import linregress
import math
import psycopg2

from app import db
from app.models.user_models import UserProfileForm, UserDataForm, UserData, TrainModelForm, TrainedModel, ErrorLog
import uuid, json, os
import datetime

def has_column(col_name, data_frame):
    cols = data_frame.column_names()
    if col_name in cols:
        return True
    return False

def safely_add_col(col_name, data_to_add, data_frame):
    cols = data_frame.column_names()
    if col_name in cols:
        data_frame.remove_column(col_name)
    sa = SArray(data=data_to_add)
    return data_frame.add_column(sa, col_name)

def save_data(my_data, name, new_frame):
    data = UserData()
    data.user_id = current_user.id
    data.project_id = my_data.project_id
    data.description = my_data.description
    data.path = my_data.path
    data.fname = my_data.fname
    data.name=name
    data.num_rows = new_frame.num_rows()
    data.num_cols = new_frame.num_columns()
    data.sname = os.path.join(data.path, str(uuid.uuid4())  + "_sframe")
    cols = new_frame.column_names()
    types = new_frame.column_types()
    stats = []
    for x in range(0, cols.__len__()):
        cdata_all = new_frame[cols[x]]
        data_frame_cleaned = new_frame.dropna(str(cols[x]), how="all")
        cdata = data_frame_cleaned[cols[x]]

        missing = round(float(cdata_all.countna())/float(len(cdata_all)), 2) * 100
        if (str(types[x].__name__) == "str"):
            stats.append({"min": "-", "max": "-", "mean": "-", "median": "-", "mode": "-", "std": "-", "var": "-", "sum": "-", "missing": missing})
        else:
            ndata = cdata.to_numpy()
            ndata = np.array(ndata).astype(np.float)
            stats.append({"min": round(cdata.min(), 2), "max": round(cdata.max(), 2), "mean": round(cdata.mean(), 2), "median": round(np.median(ndata), 4), "mode": round(scipy_stats.mode(ndata).mode[0], 4), "std": round(cdata.std(), 2), "var": round(cdata.var(), 2), "sum": cdata.sum(), "missing": missing})

    data.stats = stats
    new_frame.save(data.sname)
    db.session.add(data)
    db.session.commit()
    return data.id

transforms_blueprint = Blueprint('transforms', __name__, template_folder='templates')

@transforms_blueprint.route('/split_page', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def split_page():
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        my_model = TrainedModel()
        form = TrainModelForm(request.form, obj=my_model)
        data_frame = tc.load_sframe(my_data.sname)

        if request.method == 'POST':
            training_set,test_set = data_frame.random_split(float(request.form['percent']),seed=0)
            save_data(my_data, request.form['train'], training_set)
            save_data(my_data, request.form['test'], test_set)

            flash('Successfully created train/test split for ' + my_data.name + '!', 'success')
            return redirect(url_for('main.my_project_page', project_id=my_data.project_id))

        return render_template('pages/data/transforms/split.html',
            my_data=my_data,
            form=form,
            data_frame=data_frame)
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@transforms_blueprint.route('/split_session_page', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def split_session_page():
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        my_model = TrainedModel()
        form = TrainModelForm(request.form, obj=my_model)
        data_frame = tc.load_sframe(my_data.sname)
        cols = []
        display_cols = []
        names=data_frame.column_names()
        types=data_frame.column_types()
        print(names)
        for x in range(0, names.__len__()):
            if (str(types[x].__name__) == "int"):
                cols.append(str(names[x]))
        print(cols)
        if request.method == 'POST':
            training_set,test_set = tc.activity_classifier.util.random_split_by_session(data_frame, session_id=str(request.form['idField']), fraction=float(request.form['percent']))
            save_data(my_data, request.form['train'], training_set)
            save_data(my_data, request.form['test'], test_set)

            flash('Successfully created train/test split for ' + my_data.name + '!', 'success')
            return redirect(url_for('main.my_project_page', project_id=my_data.project_id))

        return render_template('pages/data/transforms/split_session.html',
            my_data=my_data,
            form=form,
            data_frame=data_frame,
            types=types,
            names=cols)
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@transforms_blueprint.route('/classify', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def classify_page():
    #try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        my_model = TrainedModel()
        form = TrainModelForm(request.form, obj=my_model)
        data_frame = tc.load_sframe(my_data.sname)
        data_frame = tc.load_sframe(my_data.sname)
        target = None
        cols = []
        display_cols = []
        names=data_frame.column_names()
        types=data_frame.column_types()

        for x in range(0, names.__len__()):
            cols.append(str(names[x]))

        if request.method == 'POST':
            target = request.form['target']
            data_frame = data_frame.dropna(str(target), how="all")
            orig_data = data_frame[str(target)]
            norig_data = orig_data.to_numpy() 
            classes = []
            for data in norig_data:
                appended = False 
                for x in range(1, int(request.form['num_brackets'])+1):
                    if float(data) >= float(request.form['lrange_' + str(x)]) and float(data) <= float(request.form['urange_' + str(x)]):
                        print(request.form['class_' + str(x)]) 
                        classes.append(request.form['class_' + str(x)])
                        appended = True
                        continue 
                if appended == False:
                    classes.append("unknown")   

            data_frame = safely_add_col(str(request.form['field']), classes, data_frame)            
            fwd_id = save_data(my_data, request.form['name'], data_frame)
  
            flash('Successfully transformed the data set!', 'success')
            return redirect(url_for('data.data_details_page', data_id=fwd_id))

        return render_template('pages/data/transforms/classifier.html',
            my_data=my_data,
            form=form,
            data_frame=data_frame,
            names=names,
            types=types,
            target=target,
            cols=cols)
    # except Exception as e:
    #     flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
    #     error = ErrorLog()
    #     error.user_id = current_user.id
    #     error.error = str(e.__class__)
    #     error.parameters = request.args
    #     db.session.add(error)
    #     db.session.commit()
    #     return redirect(request.referrer)

@transforms_blueprint.route('/dedup', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def dedup_page():
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        my_model = TrainedModel()
        form = TrainModelForm(request.form, obj=my_model)
        data_frame = tc.load_sframe(my_data.sname)

        if request.method == 'POST':
            data_frame = data_frame.unique()
            fwd_id = save_data(my_data, request.form['name'], data_frame)

            flash('Successfully transformed the data set!', 'success')
            return redirect(url_for('data.data_details_page', data_id=fwd_id))

        return render_template('pages/data/transforms/dedup.html',
            my_data=my_data,
            form=form,
            data_frame=data_frame)
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@transforms_blueprint.route('/sample_page', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def sample_page():
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        my_model = TrainedModel()
        form = TrainModelForm(request.form, obj=my_model)
        data_frame = tc.load_sframe(my_data.sname)

        if request.method == 'POST':
            data_frame = data_frame.sample(float(request.form['percent']))
            fwd_id = save_data(my_data, request.form['name'], data_frame)

            flash('Successfully sampled ' + my_data.name + '!', 'success')
            return redirect(url_for('data.data_details_page', data_id=fwd_id))

        return render_template('pages/data/transforms/sample.html',
            my_data=my_data,
            form=form,
            data_frame=data_frame)
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@transforms_blueprint.route('/recode_step1', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def recode_step1_page():
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        my_model = TrainedModel()
        form = TrainModelForm(request.form, obj=my_model)
        data_frame = tc.load_sframe(my_data.sname)
        target = None
        cols = []
        display_cols = []
        names=data_frame.column_names()
        types=data_frame.column_types()

        for x in range(0, names.__len__()):
            if (str(types[x].__name__) == "str"):
                cols.append(str(names[x]))

        if request.method == 'POST':
            target = request.form['target']
            return redirect(url_for('transforms.recode_step2_page', data_id=my_data.id, target=target, name=request.form['name']))

        return render_template('pages/data/transforms/code_field.html',
            my_data=my_data,
            form=form,
            data_frame=data_frame,
            names=names,
            types=types,
            target=target,
            cols=cols)
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@transforms_blueprint.route('/fill_na', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def fill_na_page():
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        my_model = TrainedModel()
        form = TrainModelForm(request.form, obj=my_model)
        data_frame = tc.load_sframe(my_data.sname)
        names=data_frame.column_names()
        types=data_frame.column_types()

        if request.method == 'POST':
            value = str(request.form['value'])
            name = str(request.form['name'])
            for feature in request.form.getlist('features'):
                orig_data = data_frame[str(feature)]
                print(orig_data.dtype.__name__)
                if orig_data.dtype.__name__ == "int":
                    try:
                        data_frame[str(feature)] = orig_data.fillna(int(value))
                    except Exception as e:
                        flash('Opps!  Looks like you passed something I could not parse as an integer.', 'error')
                        return redirect(request.referrer)
                if orig_data.dtype.__name__ == "float":
                    try:
                        data_frame[str(feature)] = orig_data.fillna(float(value))
                    except Exception as e:
                        flash('Opps!  Looks like you passed something I could not parse as an float.', 'error')
                        return redirect(request.referrer)
                if orig_data.dtype.__name__ == "str":
                    try:
                        data_frame[str(feature)] = orig_data.fillna(str(value))
                    except Exception as e:
                        flash('Opps!  Looks like you passed something I could not parse as an string.', 'error')
                        return redirect(request.referrer)
            fwd_id = save_data(my_data, name, data_frame)
            flash('Successfully replaced N/A values!', 'success')
            return redirect(url_for('data.data_details_page', data_id=fwd_id))

        return render_template('pages/data/transforms/fill_na.html',
            my_data=my_data,
            data_frame=data_frame,
            names=names,
            types=types,
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

@transforms_blueprint.route('/outlliers', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def outlliers_page():
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        my_model = TrainedModel()
        form = TrainModelForm(request.form, obj=my_model)
        data_frame = tc.load_sframe(my_data.sname)
        names=data_frame.column_names()
        types=data_frame.column_types()

        if request.method == 'POST':
            cent = float(request.form['cent'])
            name = str(request.form['name'])
            target = str(request.form['target'])
            mean = data_frame[target].mean()
            rows = []
            for row in data_frame:
                if row[target] is not None:
                    diff = abs(float(row[target]) - mean)
                    pdiff = diff/mean
                    if pdiff < cent:
                        rows.append(row)
                else:
                    rows.append(row)
            sf = tc.SFrame(rows)
            sf = sf.unpack('X1', column_name_prefix='')
            print(sf)
            fwd_id = save_data(my_data, name, sf)
            flash('Successfully removed outliers!', 'success')
            return redirect(url_for('data.data_details_page', data_id=fwd_id))
        return render_template('pages/data/transforms/outlier.html',
            my_data=my_data,
            data_frame=data_frame,
            names=names,
            types=types,
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

@transforms_blueprint.route('/rename_feature', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def rename_feature_page():
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        my_model = TrainedModel()
        form = TrainModelForm(request.form, obj=my_model)
        data_frame = tc.load_sframe(my_data.sname)
        names=data_frame.column_names()
        types=data_frame.column_types()

        if request.method == 'POST':
            feature_name = str(request.form['feature_name'])
            name = str(request.form['name'])
            target = str(request.form['target'])
            if has_column(feature_name, data_frame):
                flash('Opps!  You appear to already have a feature with this name.', 'error')
                return redirect(request.referrer)
            sf = data_frame.rename({target: feature_name})
            fwd_id = save_data(my_data, name, sf)
            flash('Successfully transformed the data!', 'success')
            return redirect(url_for('data.data_details_page', data_id=fwd_id))
        return render_template('pages/data/transforms/rename_feature.html',
            my_data=my_data,
            data_frame=data_frame,
            names=names,
            types=types,
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

@transforms_blueprint.route('/unique', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def unique_page():
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        my_model = TrainedModel()
        form = TrainModelForm(request.form, obj=my_model)
        data_frame = tc.load_sframe(my_data.sname)

        if request.method == 'POST':
            new_id = str(request.form['new_id'])
            name = str(request.form['name'])
            sf = data_frame.add_row_number(new_id)
            fwd_id = save_data(my_data, name, sf)
            flash('Successfully transformed the data!', 'success')
            return redirect(url_for('data.data_details_page', data_id=fwd_id))
        return render_template('pages/data/transforms/unique.html',
            my_data=my_data,
            data_frame=data_frame,
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

@transforms_blueprint.route('/outlliers_threshold', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def outlliers_threshold_page():
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        my_model = TrainedModel()
        form = TrainModelForm(request.form, obj=my_model)
        data_frame = tc.load_sframe(my_data.sname)
        names=data_frame.column_names()
        types=data_frame.column_types()

        if request.method == 'POST':
            threshold = float(request.form['threshold'])
            name = str(request.form['name'])
            target = str(request.form['target'])
            mean = data_frame[target].mean()
            rows = []
            for row in data_frame:
                if row[target] is not None:
                    if row[target] < threshold:
                        rows.append(row)
                else:
                    rows.append(row)
            sf = tc.SFrame(rows)
            sf = sf.unpack('X1', column_name_prefix='')
            print(sf)
            fwd_id = save_data(my_data, name, sf)
            flash('Successfully removed outliers!', 'success')
            return redirect(url_for('data.data_details_page', data_id=fwd_id))
        return render_template('pages/data/transforms/outlier_threshold.html',
            my_data=my_data,
            data_frame=data_frame,
            names=names,
            types=types,
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

@transforms_blueprint.route('/convert_magic', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def convert_magic_page():
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        my_model = TrainedModel()
        form = TrainModelForm(request.form, obj=my_model)
        data_frame = tc.load_sframe(my_data.sname)
        names=data_frame.column_names()
        types=data_frame.column_types()

        if request.method == 'POST':
            magic = str(request.form['magic'])
            name = str(request.form['name'])
            for feature in request.form.getlist('features'):
                orig_data = data_frame[str(feature)]
                norig_data = orig_data.to_numpy()
                new_data = []
                for item in norig_data:
                    if str(item) == magic:
                        new_data.append(None)
                    else:
                        new_data.append(item)
                sa = SArray(new_data)
                data_frame[str(feature)] = sa
            fwd_id = save_data(my_data, name, data_frame)
            flash('Successfully cleared magic values!', 'success')
            return redirect(url_for('data.data_details_page', data_id=fwd_id))

        return render_template('pages/data/transforms/convert_magic.html',
            my_data=my_data,
            data_frame=data_frame,
            names=names,
            types=types,
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

@transforms_blueprint.route('/recode_step2', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def recode_step2_page():
    try:
        data_id = request.args.get('data_id')
        target = request.args.get('target')
        name = request.args.get('name')
        my_data = UserData.query.filter_by(id=data_id).first()
        my_model = TrainedModel()
        form = TrainModelForm(request.form, obj=my_model)
        data_frame = tc.load_sframe(my_data.sname)
        names=data_frame.column_names()
        types=data_frame.column_types()

        orig_data = data_frame[str(target)]
        norig_data = orig_data.to_numpy()

        target_data = data_frame[str(target)].unique()
        ntarget_data = target_data.to_numpy()

        if request.method == 'POST':
            mapped_values = []
            data_frame = safely_add_col(str(target) + '_uncoded', data_frame[str(target)], data_frame)
            for x in range(0, ntarget_data.__len__()):
                mapped_values.append(str(request.form['new_value' + str(x)]))
            cross_ref = []
            for x in range(0, names.__len__()):
                if (str(types[x].__name__) == "str"):
                    cross_ref.append(str(names[x]))
            new_data = []
            for field in norig_data:
                for y in range(0, ntarget_data.__len__()):
                    if str(ntarget_data[y]) == str(field):
                        new_data.append(int(mapped_values[y]))
            sa = SArray(new_data)
            data_frame[str(target)] = sa
            fwd_id = save_data(my_data, name, data_frame)

            flash('Successfully re-coded ' + target + '!', 'success')
            return redirect(url_for('data.data_details_page', data_id=fwd_id))
        return render_template('pages/data/transforms/code_field_step2.html',
            my_data=my_data,
            form=form,
            data_frame=data_frame,
            names=names,
            name=name,
            types=types,
            ntarget_data=ntarget_data,
            target=target)
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@transforms_blueprint.route('/remove_columns', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def remove_columns_page():
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        my_model = TrainedModel()
        form = TrainModelForm(request.form, obj=my_model)
        data_frame = tc.load_sframe(my_data.sname)

        if request.method == 'POST':
            features_utf = request.form.getlist('features')
            features_str = []

            for feat in features_utf:
                features_str.append(str(feat))
            sframe = data_frame.remove_columns(features_str)
            fwd_id = save_data(my_data, request.form['name'], sframe)

            flash('Data transform is sucessful!', 'success')
            return redirect(url_for('data.data_details_page', data_id=fwd_id))
        return render_template('pages/data/transforms/remove_columns.html',
            my_data=my_data,
            form=form,
            data_frame=data_frame,
            names=data_frame.column_names(),
            types=data_frame.column_types())
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@transforms_blueprint.route('/smote', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def smote_page():
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        my_model = TrainedModel()
        form = TrainModelForm(request.form, obj=my_model)
        data_frame = tc.load_sframe(my_data.sname)

        if request.method == 'POST':
            features_utf = request.form.getlist('features')
            features_str = []

            for feat in features_utf:
                features_str.append(str(feat))
            sframe = data_frame.remove_columns(features_str)
            fwd_id = save_data(my_data, request.form['name'], sframe)

            flash('Data transform is sucessful!', 'success')
            return redirect(url_for('data.data_details_page', data_id=fwd_id))
        return render_template('pages/data/transforms/smote.html',
            my_data=my_data,
            form=form,
            data_frame=data_frame,
            names=data_frame.column_names(),
            types=data_frame.column_types())
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@transforms_blueprint.route('/calculate_slope', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def calculate_slope_page():
    # try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        my_model = TrainedModel()
        form = TrainModelForm(request.form, obj=my_model)
        data_frame = tc.load_sframe(my_data.sname)

        if request.method == 'POST':
            idField = request.form['idField']
            target = request.form['target']
            xVal = request.form['xVal']
            featArrays = {}
            features_utf = request.form.getlist('features')
            features_str = []

            for feat in features_utf:
                features_str.append(str(feat))
            for feature in request.form.getlist('features'):
                featArrays[str(feature)] = []
            featArrays[xVal] = []

            for feature in request.form.getlist('features'):
                data_frame = data_frame.dropna(str(feature), how="all")
            data_frame = data_frame.dropna(str(idField), how="all")
            data_frame = data_frame.dropna(str(xVal), how="all")
            size = len(data_frame)

            # Setup the final hash - make sure we have keys for everything
            final_frame_vals = {}
            for name in data_frame.column_names():
                if name not in features_str:
                    final_frame_vals[str(name)] = []
            for key, value in featArrays.items():
                if key in features_str:
                    final_frame_vals[str(key) + "_slope"] = []
                    final_frame_vals[str(key) + "_intercept"] = []
            final_frame_vals[str(target) + "_initial"] = []
            targetVal = None
            for x in range(0, size):
                row = data_frame[x]
                if x == 0 or data_frame[x-1][idField] != row[idField]:
                    targetVal = row[target]
                    print(targetVal)
                #print(row)
                for feature in request.form.getlist('features'):
                    featArrays[str(feature)].append(row[str(feature)])
                featArrays[xVal].append(row[str(xVal)])
                if x == size-1 or row[idField] != data_frame[x+1][idField]:
                    finalRow = {}
                    for name in data_frame.column_names():
                        if (str(name) not in features_str):
                            finalRow[str(name)] = row[str(name)]
                    xValArr = np.array(featArrays[xVal]).astype(np.float)
                    for key, value in featArrays.items():
                        if key != xVal:
                            yValArr = np.array(value).astype(np.float)
                            slope, intercept, r_value, p_value, std_err = linregress(xValArr, yValArr)
                            finalRow[key+"_slope"] = float(slope)
                            finalRow[key+"_intercept"] = float(intercept)
                            if np.isnan(float(slope)) or np.isnan(float(intercept)):
                                print("Got me a nan")
                    print(finalRow)
                    for key, value in finalRow.items():
                        final_frame_vals[str(key)].append(value)
                    final_frame_vals[str(target) + "_initial"].append(targetVal)
                    # Clear slope and static Xval accumulators
                    for feature in request.form.getlist('features'):
                        featArrays[str(feature)] = []
                    featArrays[str(xVal)] = []
            sframe = tc.SFrame(data=final_frame_vals)
            fwd_id = save_data(my_data, request.form['name'], sframe)

            flash('Data transform is sucessful!', 'success')
            return redirect(url_for('data.data_details_page', data_id=fwd_id))
        return render_template('pages/data/transforms/calculate_slope.html',
            my_data=my_data,
            form=form,
            data_frame=data_frame,
            names=data_frame.column_names(),
            types=data_frame.column_types())
    # except Exception as e:
    #     flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
    #     error = ErrorLog()
    #     error.user_id = current_user.id
    #     error.error = str(e.__class__)
    #     error.parameters = request.args
    #     db.session.add(error)
    #     db.session.commit()
    #     return redirect(request.referrer)

@transforms_blueprint.route('/rolling_slope', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def rolling_slope_page():
    # try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        my_model = TrainedModel()
        form = TrainModelForm(request.form, obj=my_model)
        data_frame = tc.load_sframe(my_data.sname)

        if request.method == 'POST':
            idField = request.form['idField']
            target = request.form['target']
            xVal = request.form['xVal']
            featArrays = {}
            features_utf = request.form.getlist('features')
            features_str = []

            for feat in features_utf:
                features_str.append(str(feat))
            for feature in request.form.getlist('features'):
                featArrays[str(feature)] = []
            featArrays[xVal] = []

            for feature in request.form.getlist('features'):
                data_frame = data_frame.dropna(str(feature), how="all")
            data_frame = data_frame.dropna(str(idField), how="all")
            data_frame = data_frame.dropna(str(xVal), how="all")
            size = len(data_frame)

            # Setup the final hash - make sure we have keys for everything
            final_frame_vals = {}
            for name in data_frame.column_names():
                if name not in features_str:
                    final_frame_vals[str(name)] = []
            for key, value in featArrays.items():
                if key in features_str:
                    final_frame_vals[str(key) + "_slope"] = []
                    final_frame_vals[str(key) + "_intercept"] = []
            final_frame_vals[str(target) + "_initial"] = []
            targetVal = None
            for x in range(0, size):
                row = data_frame[x]
                if x == 0 or data_frame[x-1][idField] != row[idField]:
                    targetVal = row[target]

                for feature in request.form.getlist('features'):
                    featArrays[str(feature)].append(row[str(feature)])
                featArrays[xVal].append(row[str(xVal)])

                if len(featArrays) > 1:
                    finalRow = {}
                    for name in data_frame.column_names():
                        if (str(name) not in features_str):
                            finalRow[str(name)] = row[str(name)]
                    xValArr = np.array(featArrays[xVal]).astype(np.float)
                    for key, value in featArrays.items():
                        if key != xVal:
                            yValArr = np.array(value).astype(np.float)
                            slope, intercept, r_value, p_value, std_err = linregress(xValArr, yValArr)
                            finalRow[key+"_slope"] = float(slope)
                            finalRow[key+"_intercept"] = float(intercept)

                    for key, value in finalRow.items():
                        final_frame_vals[str(key)].append(value)
                    final_frame_vals[str(target) + "_initial"].append(targetVal)
                if x == size-1 or row[idField] != data_frame[x+1][idField]:
                    # Clear slope and static Xval accumulators
                    for feature in request.form.getlist('features'):
                        featArrays[str(feature)] = []
                    featArrays[str(xVal)] = []
            sframe = tc.SFrame(data=final_frame_vals)
            fwd_id = save_data(my_data, request.form['name'], sframe)

            flash('Data transform is sucessful!', 'success')
            return redirect(url_for('data.data_details_page', data_id=fwd_id))
        return render_template('pages/data/transforms/rolling_slope.html',
            my_data=my_data,
            form=form,
            data_frame=data_frame,
            names=data_frame.column_names(),
            types=data_frame.column_types())
    # except Exception as e:
    #     flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
    #     error = ErrorLog()
    #     error.user_id = current_user.id
    #     error.error = str(e.__class__)
    #     error.parameters = request.args
    #     db.session.add(error)
    #     db.session.commit()
    #     return redirect(request.referrer)

@transforms_blueprint.route('/custom_transform', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def custom_transform_page():
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        form = UserDataForm(request.form, obj=None)
        data_frame = tc.load_sframe(my_data.sname)
        cols = data_frame.column_names()
        types = data_frame.column_types()

        if request.method == 'POST':
            try:
                transform_code = request.form['transform_code']
                target = request.form['target']
                name = request.form['name']
                transformed_data = []
                local_space = {}
                for val in data_frame[str(target)]:
                    context = {"in_var": 1, "scipy_stats": scipy_stats, "np": np}
                    exec(transform_code) in context
                    transformed_data.append(context['out_var'])
                sa = SArray(transformed_data)
                data_frame[str(name)] = sa
                fwd_id = save_data(my_data, request.form['name'], data_frame)

                flash('Data transform is sucessful!', 'success')
                return redirect(url_for('data.data_details_page', data_id=fwd_id))
            except Exception as inst:
                flash('Failed to run Python code! ' + str(inst), 'error')
        return render_template('pages/data/transforms/custom_transform.html',
            my_data=my_data,
            names=cols,
            types=types,
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

@transforms_blueprint.route('/transform')
@login_required  # Limits access to authenticated users
def transform_page():
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()

        data_frame = tc.load_sframe(my_data.sname)
        cols = data_frame.column_names()
        types = data_frame.column_types()

        return render_template('pages/data/transforms/transform.html',
            my_data=my_data,
            data_frame=data_frame.head(),
            names=cols,
            types=types)
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)
