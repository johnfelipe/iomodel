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
import sys
import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import linregress
import math
import random
import psycopg2

from app import db
from app.models.user_models import UserProfileForm, UserDataForm, UserData, TrainModelForm, TrainedModel, ErrorLog, StorageSlice, ClusterAnalysis
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
    data.description = "Cluster analysis based on " + my_data.name
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

data_blueprint = Blueprint('data', __name__, template_folder='templates')

@data_blueprint.route('/data/<path:filename>', methods=['GET'])
def csv_download(filename):
    # try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        data_frame = tc.load_sframe(my_data.sname)
        if not os.path.isfile(my_data.sname + "_export.csv"):   
            data_frame.export_csv(my_data.sname + "_export.csv")    
        root_dir = os.path.dirname(os.getcwd())
        direc = os.path.dirname(my_data.path)
        direc = os.path.join(direc, str(my_data.user_id))
        print(current_app.config['APP_FOLDER'] + direc)
        print(os.path.basename(my_data.sname) + "_export.csv")
        return send_from_directory(directory=direc, filename=os.path.basename(my_data.sname) + "_export.csv")
    # except Exception as e:
    #     flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
    #     error = ErrorLog()
    #     error.user_id = current_user.id
    #     error.error = str(e.__class__)
    #     error.parameters = request.args
    #     db.session.add(error)
    #     db.session.commit()
    #     return redirect(request.referrer)


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

@data_blueprint.route('/delete_cluster', methods=['GET'])
@login_required  # Limits access to authenticated users
def delete_cluster():
    try:
        cluster_id = request.args.get('cluster_id')
        data_id = request.args.get('data_id')

        db.session.query(ClusterAnalysis).filter_by(id = cluster_id).delete()
        db.session.commit()

        flash('You successfully deleted your cluster analysis!', 'success')
        return redirect(url_for('data.cluster_page', data_id=data_id))
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

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

@data_blueprint.route('/compare', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def compare_page():
        data_id = request.args.get('data_id')
    # try:
        my_data = UserData.query.filter_by(id=data_id).first()
        if my_data.user_id is not current_user.id:
            flash('Opps!  Do data found', 'error')
            return redirect(request.referrer)
        form = TrainModelForm(request.form, obj=None)
        data_frame = tc.load_sframe(my_data.sname)
        render_plot = False
        target = None
        cols = []
        display_cols = []
        features = []
        categories = []
        plot_data = []
        means = {}
        boxplots = {}
        outliers = {}        
        total_attrs = 0
        names=data_frame.column_names()
        types=data_frame.column_types()
        num_matching = 0
        total_rows = 0

        if request.method == 'POST':
            render_plot = True
            truth = request.form['truth']
            comp = request.form['comp']
            features = request.form.getlist('features')
            data_frame = tc.load_sframe(my_data.sname)
            data_frame = data_frame.dropna(str(truth), how="any")
            data_frame = data_frame.dropna(str(comp), how="any")
            for x in range(0, names.__len__()):
                if (str(names[x]) != truth and str(names[x]) != comp and str(types[x].__name__) != "str"):
                    cols.append(str(names[x]))
                    data_frame = data_frame.dropna(str(names[x]), how="any")
            total_rows = data_frame.num_rows()
            
            for row in data_frame:
                if (str(row[truth]) == str(row[comp])):
                    num_matching = num_matching + 1

            labels = data_frame[truth].unique()
            for feature in features:
                current_data = data_frame[str(feature)]
                means[feature] = current_data.mean()
                lbl_arry = []
                label_outliers = []
                index = 0
                for label in labels:
                    sub = data_frame[(data_frame[truth] == label)]
                    current_data = sub[str(feature)]
                    ncdata = current_data.to_numpy()
                    upper = np.percentile(ncdata,75)
                    lower = np.percentile(ncdata,25)
                    for item in ncdata:
                        if item > upper or item < lower:
                            label_outliers.append([int(index), item])                    
                    lbl_arry.append([round(np.nanmin(ncdata), 2), lower, round(np.nanmean(ncdata), 2), upper, round(np.nanmax(ncdata), 2)])

                    sub = data_frame[(data_frame[comp] == label)]
                    current_data = sub[str(feature)]
                    ncdata = current_data.to_numpy()
                    upper = np.percentile(ncdata,75)
                    lower = np.percentile(ncdata,25)
                    for item in ncdata:
                        if item > upper or item < lower:
                            label_outliers.append([int(index), item])                    
                    lbl_arry.append([round(np.nanmin(ncdata), 2), lower, round(np.nanmean(ncdata), 2), upper, round(np.nanmax(ncdata), 2)])

                outliers[feature] = label_outliers
                boxplots[feature] = lbl_arry   
            for label in labels:    
                categories.append(str(label) + "_truth")   
                categories.append(str(label) + "_comp")    
        return render_template('pages/data/compare.html',
            my_data=my_data,
            form=form,
            data_frame=data_frame,
            names=names,
            render_plot=render_plot,
            features=features,
            means=means,
            boxplots=boxplots,
            outliers=outliers,
            labels=categories,
            correct=num_matching,
            incorrect=total_rows-num_matching,
            types=types,
            target=target,
            cols=cols,
            display_cols=display_cols)
    # except Exception as e:
    #     flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
    #     error = ErrorLog()
    #     error.user_id = current_user.id
    #     error.error = str(e.__class__)
    #     error.parameters = request.args
    #     db.session.add(error)
    #     db.session.commit()
    #     return redirect(request.referrer)

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

@data_blueprint.route('/slice', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def slice_page():
    # try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        data_frame = tc.load_sframe(my_data.sname)
        form = TrainModelForm(request.form, obj=None)
        means = {}
        boxplots = {}
        outliers = {}
        distribution = []
        labels = []
        features = []
        to_render = None
        cols = data_frame.column_names()
        types = data_frame.column_types()  

        if my_data.user_id is not current_user.id:
            flash('Opps!  Do data found', 'error')
            return redirect(request.referrer)

        if request.method == 'POST':
            to_render = True
            data_frame_cleaned = data_frame
            for feature in request.form.getlist('features'):
                data_frame_cleaned = data_frame_cleaned.dropna(str(feature), how="any")
            data_frame_cleaned = data_frame_cleaned.dropna(str(request.form['label']), how="any")    
            if data_frame_cleaned.num_rows() < 2:
                flash('After cleaning, there is no data left. You have a data quality issue.', 'error')
                return redirect(url_for('data.data_details_page', data_id=data_id))       
                            
            features = request.form.getlist('features')
            target = request.form['label']
            labels = data_frame_cleaned[target].unique()
            for label in labels:
                distribution.append({"name": label, "y": 0})
            for row in data_frame:
                for item in distribution:
                    if row[target] == item['name']:
                        item['y'] = item['y'] + 1

            for feature in features:
                current_data = data_frame_cleaned[str(feature)]
                means[feature] = current_data.mean()
                lbl_arry = []
                label_outliers = []
                index = 0
                for label in labels:                    
                    sub = data_frame_cleaned[(data_frame_cleaned[target] == label)]
                    current_data = sub[str(feature)]
                    ncdata = current_data.to_numpy()
                    upper = np.percentile(ncdata,75)
                    lower = np.percentile(ncdata,25)
                    for item in ncdata:
                        if item > upper or item < lower:
                            label_outliers.append([int(index), item])                    
                    lbl_arry.append([round(np.nanmin(ncdata), 2), lower, round(np.nanmean(ncdata), 2), upper, round(np.nanmax(ncdata), 2)])
                    index = index + 1
                outliers[feature] = label_outliers
                boxplots[feature] = lbl_arry

        return render_template('pages/data/slice.html',
            my_data=my_data,
            data_frame=data_frame,
            labels=labels,
            form=form,
            features=features,
            to_render=to_render,
            means=means,
            outliers=outliers,
            boxplots=boxplots,
            distribution=distribution,
            names=cols,
            types=types)
    # except Exception as e:
    #     flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
    #     error = ErrorLog()
    #     error.user_id = current_user.id
    #     error.error = str(e.__class__)
    #     error.parameters = request.args
    #     db.session.add(error)
    #     db.session.commit()
    #     return redirect(request.referrer)

@data_blueprint.route('/view_cluster')
@login_required  # Limits access to authenticated users
def view_cluster_page():
    # try:
        cluster_id = request.args.get('cluster_id')
        clstr = ClusterAnalysis.query.filter_by(id=cluster_id).first()
        my_data = UserData.query.filter_by(id=clstr.data_id).first()
        derived_data = UserData.query.filter_by(id=clstr.derived_data_id).first()
        
        if my_data.user_id is not current_user.id:
            flash('Opps!  Do data found', 'error')
            return redirect(request.referrer)


        data_frame = tc.load_sframe(derived_data.sname)
        labels = data_frame['label'].unique()
        distribution = []
        for label in labels:
            distribution.append({"name": label, "y": 0})
        for row in data_frame:
            for item in distribution:
                if row['label'] == item['name']:
                    item['y'] = item['y'] + 1
        cols = data_frame.column_names()
        types = data_frame.column_types()

        means = {}
        boxplots = {}
        outliers = {}
        for feature in clstr.params["features"]:
            current_data = data_frame[str(feature)]
            means[feature] = current_data.mean()
            lbl_arry = []
            label_outliers = []
            index = 0
            for label in labels:
                sub = data_frame[(data_frame['label'] == label)]
                current_data = sub[str(feature)]
                ncdata = current_data.to_numpy()
                upper = np.percentile(ncdata,75)
                lower = np.percentile(ncdata,25)
                for item in ncdata:
                    if item > upper or item < lower:
                        label_outliers.append([int(index), item])                    
                lbl_arry.append([round(np.nanmin(ncdata), 2), lower, round(np.nanmean(ncdata), 2), upper, round(np.nanmax(ncdata), 2)])
                index = index + 1
            boxplots[feature] = lbl_arry
            outliers[feature] = label_outliers

        return render_template('pages/data/clusters/cluster.html',
            my_data=my_data,
            data_frame=data_frame,
            clstr=clstr,
            labels=labels,
            derived_data=derived_data,
            means=means,
            boxplots=boxplots,
            outliers=outliers,
            distribution=distribution,
            names=cols,
            types=types)
    # except Exception as e:
    #     flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
    #     error = ErrorLog()
    #     error.user_id = current_user.id
    #     error.error = str(e.__class__)
    #     error.parameters = request.args
    #     db.session.add(error)
    #     db.session.commit()
    #     return redirect(request.referrer)

@data_blueprint.route('/cluster')
@login_required  # Limits access to authenticated users
def cluster_page():
    try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        clusters = ClusterAnalysis.query.filter_by(data_id=data_id).all()
        
        if my_data.user_id is not current_user.id:
            flash('Opps!  Do data found', 'error')
            return redirect(request.referrer)

        data_frame = tc.load_sframe(my_data.sname)
        cols = data_frame.column_names()
        types = data_frame.column_types()

        return render_template('pages/data/clusters/clusters.html',
            my_data=my_data,
            data_frame=data_frame,
            clusters=clusters,
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

@data_blueprint.route('/new_cluster', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def new_cluster():
    # try:
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        if my_data.user_id is not current_user.id:
            flash('Opps!  Do data found', 'error')
            return redirect(request.referrer)
        form = UserDataForm(request.form, obj=None)
        data_frame = tc.load_sframe(my_data.sname)
        cols = data_frame.column_names()
        types = data_frame.column_types()

        if request.method == 'POST':
            data_frame_cleaned = data_frame
            cols = []
            for feature in request.form.getlist('features'):
                data_frame_cleaned = data_frame_cleaned.dropna(str(feature), how="any")
                cols.append(str(feature))
            if data_frame_cleaned.num_rows() < 2:
                flash('After cleaning, there is no data left. You have a data quality issue.', 'error')
                return redirect(url_for('data.new_cluster', data_id=data_id))       

            my_model = None
            if request.form['model_type'] == "dbscan":
                radius = 1.0
                min_core_neighbors = 10
                if request.form['radius'] is not None:
                    radius = request.form['radius']
                if request.form['min_core_neighbors'] is not None:
                    min_core_neighbors = request.form['min_core_neighbors']
                my_model = tc.dbscan.create(data_frame_cleaned, radius=float(radius), min_core_neighbors=int(min_core_neighbors), features=cols)
                my_model.summary()

                values = [None]*data_frame_cleaned.num_rows()
                types = [None]*data_frame_cleaned.num_rows()
                for row in my_model.cluster_id:
                    values[row['row_id']] = int(row['cluster_id'])
                    types[row['row_id']] = row['type']

                clstr = ClusterAnalysis()
                clstr.project_id = my_data.project_id
                clstr.user_id = current_user.id
                clstr.name = request.form['name']
                clstr.cluster_type = request.form['model_type']
                clstr_data = safely_add_col("label", values, data_frame_cleaned)
                clstr_data = safely_add_col("type", types, clstr_data)
                clstr.data_id = data_id
                clstr.derived_data_id = save_data(my_data, str(request.form['name']) + " cluster output", clstr_data)
                clstr.params = {"min_core_neighbors": int(min_core_neighbors), "radius": float(radius), "features": cols}
                db.session.add(clstr)
                db.session.commit()                   
                flash('Cluster analysis complete!', 'success')
                return redirect(url_for('data.cluster_page', data_id=data_id))
                              
            else:    
                num_clusters = 2
                max_iterations = 10
                if request.form['num_clusters'] is not None:
                    num_clusters = request.form['num_clusters']
                if request.form['max_iterations'] is not None:
                    max_iterations = request.form['max_iterations']
                my_model = tc.kmeans.create(data_frame_cleaned, num_clusters=int(num_clusters), max_iterations=int(max_iterations), features=cols)            
                my_model.summary()

                values = [None]*data_frame_cleaned.num_rows()
                distances = [None]*data_frame_cleaned.num_rows()
                for row in my_model.cluster_id:
                    values[row['row_id']] = int(row['cluster_id'])
                    distances[row['row_id']] = row['distance']

                clstr = ClusterAnalysis()
                clstr.project_id = my_data.project_id
                clstr.user_id = current_user.id
                clstr.name = request.form['name']
                clstr.cluster_type = request.form['model_type']
                clstr_data = safely_add_col("label", values, data_frame_cleaned)
                clstr_data = safely_add_col("distance", distances, clstr_data)
                clstr.data_id = data_id
                clstr.derived_data_id = save_data(my_data, str(request.form['name']) + " cluster output", clstr_data)
                clstr.params = {"num_clusters": int(num_clusters), "max_iterations": int(max_iterations), "features": cols}
                db.session.add(clstr)
                db.session.commit()   
                flash('Cluster analysis complete!', 'success')
                return redirect(url_for('data.cluster_page', data_id=data_id))

        return render_template('pages/data/clusters/new.html',
            my_data=my_data,
            data_frame=data_frame,
            form=form,
            names=cols,
            types=types)
    # except Exception as e:
    #     flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
    #     error = ErrorLog()
    #     error.user_id = current_user.id
    #     error.error = str(e.__class__)
    #     error.parameters = request.args
    #     db.session.add(error)
    #     db.session.commit()
    #     return redirect(request.referrer)

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
