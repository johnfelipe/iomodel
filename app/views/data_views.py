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
from cStringIO import StringIO
import sys
import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import linregress
import math
import random
import psycopg2

from app import db
from app.models.user_models import UserProfileForm, UserDataForm, UserData, TrainModelForm, TrainedModel, ErrorLog, StorageSlice
import uuid, json, os
import datetime

data_blueprint = Blueprint('data', __name__, template_folder='templates')

@data_blueprint.route('/data/<path:filename>', methods=['GET'])
def csv_download(filename):
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        data_frame = tc.load_sframe(my_data.sname)
        print(my_data.sname)
        if not os.path.isfile(my_data.sname + "_export.csv"):   
            data_frame.export_csv(my_data.sname + "_export.csv")    
        root_dir = os.path.dirname(os.getcwd())
        direc = os.path.dirname(my_data.path)
        direc = os.path.join(direc, str(my_data.user_id))
        return send_from_directory(directory=direc, filename=os.path.basename(my_data.sname) + "_export.csv")
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def find_slice():
    slices = db.session.query(StorageSlice).all()
    sslice = random.randint(0, len(slices)-1)
    return slices[sslice].name

def nan_to_null(f,
        _NULL=psycopg2.extensions.AsIs('NULL'),
        _NaN=np.NaN,
        _Float=psycopg2.extensions.Float):
    if not np.isnan(f):
        return f
    return None

psycopg2.extensions.register_adapter(float, nan_to_null)

@data_blueprint.route('/delete_data', methods=['GET'])
@login_required  # Limits access to authenticated users
def delete_data_page():
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        project_id = my_data.project_id

        model_count = TrainedModel.query.filter_by(data_id=data_id).count()
        if model_count > 0:
            flash('Opps! It appears that you have models that were trained with this data. Delete the models first.', 'error')
            return redirect(request.referrer)
        db.session.query(UserData).filter_by(id = data_id).delete()
        db.session.commit()

        flash('You successfully deleted your data!', 'success')
        return redirect(url_for('main.my_project_page', project_id=project_id))
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@data_blueprint.route('/data_home')
@login_required  # Limits access to authenticated users
def data_home_page():
    try:
        my_data = UserData.query.filter_by(user_id=current_user.id).all()

        return render_template('pages/data/data_page.html',
            my_data=my_data)
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@data_blueprint.route('/analyze', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def analyze_page():
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        if my_data.user_id is not current_user.id:
            flash('Opps!  Do data found', 'error')
            return redirect(request.referrer)
        my_model = TrainedModel()
        form = TrainModelForm(request.form, obj=my_model)
        data_frame = tc.load_sframe(my_data.sname)
        render_plot = False
        target = None
        cols = []
        display_cols = []
        coifs = []
        names=data_frame.column_names()
        types=data_frame.column_types()

        if request.method == 'POST':
            render_plot = True
            target = request.form['target']
            data_frame = tc.load_sframe(my_data.sname)
            data_frame = data_frame.dropna(str(target), how="any")
            for x in range(0, names.__len__()):
                if (str(names[x]) != target) and (str(types[x].__name__) != "str"):
                    cols.append(str(names[x]))
                    data_frame = data_frame.dropna(str(names[x]), how="any")
            target_data = data_frame[str(target)]
            ntarget_data = target_data.to_numpy()

            for x in range(0, names.__len__()):
                print(types[x])
                if (str(names[x]) != target) and (str(types[x].__name__) != "str"):
                    data = data_frame[str(names[x])]
                    ndata = data.to_numpy()
                    corr = np.corrcoef(ntarget_data, ndata)[0,1]
                    if math.isnan(corr):
                        coifs.append(0)
                    else:
                        coifs.append(corr)

        return render_template('pages/data/analyze.html',
            my_data=my_data,
            form=form,
            data_frame=data_frame,
            names=names,
            types=types,
            render_plot=render_plot,
            target=target,
            cols=cols,
            display_cols=display_cols,
            coifs=coifs)
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@data_blueprint.route('/scatter_analysis', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def scatter_analysis_page():
    data_id = request.args.get('data_id')
    try:
        my_data = UserData.query.filter_by(id=data_id).first()
        if my_data.user_id is not current_user.id:
            flash('Opps!  Do data found', 'error')
            return redirect(request.referrer)
        my_model = TrainedModel()
        form = TrainModelForm(request.form, obj=my_model)
        data_frame = tc.load_sframe(my_data.sname)
        render_plot = False
        target = None
        cols = []
        display_cols = []
        plot_data = []
        total_attrs = 0
        names=data_frame.column_names()
        types=data_frame.column_types()

        if request.method == 'POST':
            render_plot = True
            target = request.form['target']
            data_frame = tc.load_sframe(my_data.sname)
            data_frame = data_frame.dropna(str(target), how="any")
            for x in range(0, names.__len__()):
                if (str(names[x]) != target) and (str(types[x].__name__) != "str"):
                    cols.append(str(names[x]))
                    data_frame = data_frame.dropna(str(names[x]), how="any")
            target_data = data_frame[str(target)]
            ntarget_data = target_data.to_numpy()

            for x in range(0, names.__len__()):
                if (names[x] != target) and (types[x].__name__ != "str"):
                    scatter = []
                    df = data_frame[names[x]]
                    for y in range(0, ntarget_data.__len__()):
                        scatter.append([ntarget_data[y], df[y]])
                    total_attrs = total_attrs + 1
                    display_cols.append(str(names[x]))
                    plot_data.append(scatter)

            print(total_attrs)
        return render_template('pages/data/scatter_analysis.html',
            my_data=my_data,
            form=form,
            data_frame=data_frame,
            names=names,
            types=types,
            render_plot=render_plot,
            target=target,
            cols=cols,
            display_cols=display_cols,
            plot_data=plot_data)
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@data_blueprint.route('/data_viz')
@login_required  # Limits access to authenticated users
def data_viz_page():
    try:
        data_id = request.args.get('data_id')
        col_name = request.args.get('col_name')
        dtype = request.args.get('type')
        my_data = UserData.query.filter_by(user_id=current_user.id).filter_by(id=data_id).first()
        if my_data.user_id is not current_user.id:
            flash('Opps!  Do data found', 'error')
            return redirect(request.referrer)
        data_frame = tc.load_sframe(my_data.sname)
        data_frame = data_frame.dropna(str(col_name), how="any")
        data = data_frame[str(col_name)]
        ndata = data.to_numpy()
        sorted_results = sorted(ndata)
        to_render = {}
        if (dtype == "str"):
            to_render = {"words": '~'.join(ndata)}
        else:
            outlier = []
            upper = np.percentile(ndata,75)
            lower = np.percentile(ndata,25)
            for item in data:
                if item > upper or item < lower:
                    outlier.append([0, item])
            to_render = {"outliers": outlier, "min": round(data.min(), 2), "max": round(data.max(), 2), "mean": round(data.mean(), 2), "median": np.median(ndata), "upper": upper, "lower": lower}
        return render_template('pages/data/data_viz.html',
            my_data=my_data,
            col_name=col_name,
            results=sorted_results,
            dtype=dtype,
            to_render=to_render)
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@data_blueprint.route('/data_details')
@login_required  # Limits access to authenticated users
def data_details_page():
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        if my_data.user_id is not current_user.id:
            flash('Opps!  Do data found', 'error')
            return redirect(request.referrer)

        data_frame = tc.load_sframe(my_data.sname)
        cols = data_frame.column_names()
        types = data_frame.column_types()

        return render_template('pages/data/data_details.html',
            my_data=my_data,
            data_frame=data_frame,
            filename=os.path.basename(my_data.fname),
            names=cols,
            types=types,
            stats=my_data.stats)
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@data_blueprint.route('/top')
@login_required  # Limits access to authenticated users
def top_page():
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        if my_data.user_id is not current_user.id:
            flash('Opps!  Do data found', 'error')
            return redirect(request.referrer)

        data_frame = tc.load_sframe(my_data.sname)
        cols = data_frame.column_names()
        types = data_frame.column_types()

        return render_template('pages/data/top.html',
            my_data=my_data,
            data_frame=data_frame,
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

@data_blueprint.route('/data_quality')
@login_required  # Limits access to authenticated users
def data_quality_page():
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        if my_data.user_id is not current_user.id:
            flash('Opps!  Do data found', 'error')
            return redirect(request.referrer)

        data_frame = tc.load_sframe(my_data.sname)
        cols = data_frame.column_names()
        types = data_frame.column_types()
        total_rows = data_frame.num_rows()
        missings = []
        goods = []
        totes = 0
        the_goods = 0
        for feature in data_frame.column_names():
            cleaned_frame = data_frame.dropna(str(feature), how="all")
            totes = totes + total_rows
            the_goods = the_goods + cleaned_frame.num_rows()
            good_percent = float(cleaned_frame.num_rows()) / float(total_rows)
            goods.append(good_percent)
            missing_percent = 1-good_percent
            missings.append(missing_percent)
        good_percent = (float(the_goods)/float(totes))
        missing_percent = 1-good_percent

        return render_template('pages/data/data_quality.html',
            my_data=my_data,
            data_frame=data_frame,
            names=cols,
            missing_percent=missing_percent,
            good_percent=good_percent,
            missings=missings,
            goods=goods,
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

@data_blueprint.route('/data_import', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def data_import_page():
    # try:
        project_id = request.args.get('project_id')
        data = UserData()
        form = UserDataForm(request.form, obj=data)

        if request.method == 'POST':
            form.populate_obj(data)
            data.user_id = current_user.id

            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part', 'error')
                return redirect(request.url)
            file = request.files['file']
            # if user does not select file, browser also
            # submit a empty part without filename
            if file.filename == '':
                flash('No selected file', 'error')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                slice_name = find_slice()
                pname = os.path.join(current_app.config['UPLOAD_FOLDER'], str(slice_name))
                if not os.path.isdir(pname):
                    os.mkdir(pname)
                pname = os.path.join(pname, str(current_user.id))
                if not os.path.isdir(pname):
                    os.mkdir(pname)
                fname = os.path.join(pname, filename)
                file.save(fname)
                sframe = tc.SFrame(data=fname)
                if sframe.num_rows() < 5:
                    flash('Opps! Looks like there is something wrong with your data file. Please check the format and try again.', 'error')
                    return render_template('pages/data/import.html',
                               form=form)
                sname = os.path.join(pname, filename + "_sframe")
                sframe.save(sname)
                data.path = pname
                data.fname = fname
                data.sname = sname
                cols = sframe.column_names()
                types = sframe.column_types()
                stats = []

                for x in range(0, cols.__len__()):
                    cdata_all = sframe[cols[x]]
                    data_frame_cleaned = sframe.dropna(str(cols[x]), how="all")
                    cdata = data_frame_cleaned[cols[x]]

                    missing = round(float(cdata_all.countna())/float(len(cdata_all)), 2) * 100
                    if (str(types[x].__name__) == "str"):
                        stats.append({"min": "-", "max": "-", "mean": "-", "median": "-", "mode": "-", "std": "-", "var": "-", "sum": "-", "missing": missing})
                    else:
                        ndata = cdata.to_numpy()
                        ndata = np.array(ndata).astype(np.float)
                        stats.append({"min": round(cdata.min(), 2), "max": round(cdata.max(), 2), "mean": round(cdata.mean(), 2), "median": round(np.median(ndata), 4), "mode": round(scipy_stats.mode(ndata).mode[0], 4), "std": round(cdata.std(), 2), "var": round(cdata.var(), 2), "sum": cdata.sum(), "missing": missing})

                data.stats = stats
                data.num_rows = sframe.num_rows()
                data.num_cols = sframe.num_columns()
                data.project_id = project_id
                db.session.add(data)
                db.session.commit()
                flash('File has been imported!', 'success')
                return redirect(url_for('data.data_details_page', data_id=data.id))
            else:
                flash('Opps! Looks like there is something wrong with your data file. Please check the format and try again.', 'error')
                return render_template('pages/data/import.html',
                           form=form)
        return render_template('pages/data/import.html',
                               project_id=project_id,
                               form=form)
    # except Exception as e:
    #     flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
    #     error = ErrorLog()
    #     error.user_id = current_user.id
    #     error.error = str(e.__class__)
    #     error.parameters = request.args
    #     db.session.add(error)
    #     db.session.commit()
    #     return redirect(request.referrer)
