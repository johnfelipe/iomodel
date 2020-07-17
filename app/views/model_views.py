#
# Copyright 2018 Twin Tech Labs
# Author: Matt Hogan
#
from flask import Blueprint, redirect, render_template, current_app
from flask import request, url_for, flash, send_from_directory, jsonify, render_template_string
from flask_user import current_user, login_required, roles_accepted
from werkzeug.utils import secure_filename
import turicreate as tc
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from turicreate import SArray
import sys
import numpy as np
from scipy import stats as scipy_stats
from sklearn.utils import shuffle
import sklearn.metrics as metrics
import scikits.bootstrap as boot
import pandas as pd
from math import sqrt
import psycopg2
from app import db
from app.models.user_models import UserProfileForm, UserDataForm, UserData, TrainModelForm, TrainedModel, Predictions, ModelRun, ErrorLog
import uuid, json, os
import datetime

model_blueprint = Blueprint('model', __name__, template_folder='templates')

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def sanitize_results(original, predictions):
    new_orig = []
    new_pred = []    

    for x in range(0, len(predictions)):
        if original[x] is not None and predictions[x] is not None:
            new_orig.append(original[x])
            new_pred.append(predictions[x])

    return {"orig": new_orig, "pred": new_pred}

def sanitize_with_variance(original, predictions, variance):
    new_orig = []
    new_pred = []  
    new_var = []    

    for x in range(0, len(predictions)):
        if original[x] is not None and predictions[x] is not None and variance[x] is not None:
            new_orig.append(original[x])
            new_pred.append(predictions[x])
            new_var.append(variance[x])

    return {"orig": new_orig, "pred": new_pred, "variance": variance}    

def compute_explained_variance(norig_data, npredicted_data):
    values = sanitize_results(norig_data, npredicted_data)
    explained = metrics.explained_variance_score(values["orig"], values["pred"])
    unexplained = 1 - explained
    return explained, unexplained

def bootstrap_ci(norig_data, npredicted_data):
    variance = []
    for x in range(0, len(npredicted_data)):
        variance.append(abs(float(norig_data[x]) - float(npredicted_data[x])))
    return np.nanmean(variance)

def compute_r2(norig_data, npredicted_data):
    ssx = 0.0
    ssy = 0.0
    sst = 0.0
    mean_x = np.nanmean(norig_data)
    mean_y = np.nanmean(npredicted_data)
    for x in range(0, len(npredicted_data)):
        ssx = ssx + abs(float(norig_data[x])-float(mean_x))**2
        ssy = ssy + abs(float(npredicted_data[x])-float(mean_y))**2
        sst = sst + abs((float(norig_data[x])-float(mean_x)) * abs(float(npredicted_data[x])-float(mean_y)))
    r = sst/sqrt((ssx*ssy))
    return r**2

def safely_add_col(col_name, data_to_add, data_frame):
    cols = data_frame.column_names()
    if col_name in cols:
        data_frame.remove_column(col_name)
    sa = SArray(data=data_to_add)
    return data_frame.add_column(sa, col_name)

def nan_to_null(f,
        _NULL=psycopg2.extensions.AsIs('NULL'),
        _NaN=np.NaN,
        _Float=psycopg2.extensions.Float):
    if f is None:
        return None
    if not np.isnan(f):
        return f
    return None

psycopg2.extensions.register_adapter(float, nan_to_null)

@model_blueprint.route('/delete_model', methods=['GET'])
@login_required  # Limits access to authenticated users
def delete_model_page():
    try:
        model_id = request.args.get('model_id')
        my_model = TrainedModel.query.filter_by(id=model_id).first()
        project_id = my_model.project_id
        db.session.query(TrainedModel).filter_by(id = model_id).delete()
        db.session.commit()

        flash('You successfully deleted your model!', 'success')
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

@model_blueprint.route('/model_data/<path:filename>', methods=['GET'])
def download(filename):
    try:
        model_id = request.args.get('model_id')
        my_model = TrainedModel.query.filter_by(id=model_id).first()
        root_dir = os.path.dirname(os.getcwd())
        direc = os.path.dirname(my_model.path)
        direc = os.path.join(direc, str(my_model.user_id))
        return send_from_directory(directory=direc, filename=str(my_model.name) + str(my_model.api_key) + "_model_cross_validation.csv")
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@model_blueprint.route('/predictions/<path:filename>', methods=['GET'])
def download_predictions(filename):
    try:
        dict_id = request.args.get('dict')
        my_dict = Predictions.query.filter_by(id=dict_id).first()
        direc = os.path.dirname(my_dict.path)
        direc = os.path.join(direc, str(my_dict.user_id))

        return send_from_directory(directory=direc, filename=os.path.basename(my_dict.oname))
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@model_blueprint.route('/web_api')
@login_required  # Limits access to authenticated users
def web_api_page():
    try:
        model_id = request.args.get('model_id')
        my_model = TrainedModel.query.filter_by(id=model_id).first()
        my_data = UserData.query.filter_by(id=my_model.data_id).first()
        if my_data.user_id is not current_user.id:
            flash('Opps!  Do data found', 'error')
            return redirect(request.referrer)
        data_frame = tc.load_sframe(my_data.sname)

        names=data_frame.column_names()
        types=data_frame.column_types()
        type_map = {}
        for x in range(0, names.__len__()):
            type_map[str(names[x])] = types[x]

        example_json = {}
        for feature in my_model.features['features']:
            example_json[feature] = type_map[feature].__name__

        return render_template('pages/models/web_api.html',
            my_data=my_data,
            type_map=type_map,
            example_json=json.dumps(example_json, sort_keys = True, indent = 4, separators = (',', ': ')),
            my_model=my_model)
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@model_blueprint.route('/bootstrap')
@login_required  # Limits access to authenticated users
def bootstrap_page():
    try:
        model_id = request.args.get('model_id')
        my_model = TrainedModel.query.filter_by(id=model_id).first()
        my_data = UserData.query.filter_by(id=my_model.data_id).first()
        if my_data.user_id is not current_user.id:
            flash('Opps!  Do data found', 'error')
            return redirect(request.referrer)

        to_render = {}
        variance_array = []
        if my_model.bootstrap is not None:
             to_render = my_model.bootstrap
        else:
            data_frame = tc.load_sframe(my_data.sname)
            data_frame = data_frame.add_row_number(column_name='process_id')
            cols = []
            data_frame_cleaned = data_frame.dropna(str(my_model.features['target']), how="any")
            for feature in my_model.features['features']:
                data_frame_cleaned = data_frame_cleaned.dropna(str(feature), how="any")
                cols.append(str(feature))

            # 1. Randomly select x% of sample and train the algorithm.
            # 2. Predict scores for the remaining 100-x% of the sample
            # 3. correlate actual and predicted scores, and square the result.
            # 4. Repeat 2000 time
            # 5. arrange all outcome in ascending order
            # 6. report the 50th observation as lower bound of CI
            # 7. report 1950th observation as upper bound CI
            variance_array = []
            r2_array = []
            df = shuffle(data_frame_cleaned.to_dataframe())
            for y in range(0, 100):
                df = shuffle(df)
            data_frame = tc.SFrame(data=df)
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            for x in range(0,2000):
                # Grab data
                df = shuffle(data_frame.to_dataframe())
                data_frame = tc.SFrame(data=df)
                training_set,test_set = data_frame.random_split(.80,seed=0)

                # Train model
                tc_model = None
                if my_model.features['model_type'] == 'gradient':
                    tc_model = tc.boosted_trees_regression.create(training_set, target=str(my_model.features['target']), validation_set=None, features=cols, max_depth = my_model.options['max_depth'], max_iterations = my_model.options['max_iterations'] )
                elif my_model.features['model_type'] == 'linear':
                    tc_model = tc.linear_regression.create(training_set, target=str(my_model.features['target']), validation_set=None, features=cols )
                elif my_model.features['model_type'] == 'decision':
                    tc_model = tc.decision_tree_regression.create(training_set, target=str(my_model.features['target']), validation_set=None, features=cols, max_depth = my_model.options['max_depth'] )
                elif my_model.features['model_type'] == 'random':
                    tc_model = tc.random_forest_regression.create(training_set, target=str(my_model.features['target']), validation_set=None, features=cols, max_depth = my_model.options['max_depth'], max_iterations = my_model.options['max_iterations'] )

                predicted_scores = tc_model.predict(test_set)
                origs = []
                for item in test_set[str(my_model.features['target'])]:
                    origs.append(item)
                norigs = np.array(origs)
                npredicted_scores = np.array(predicted_scores)
                mean_var = bootstrap_ci(norigs, npredicted_scores)
                r2 = compute_r2(norigs, npredicted_scores)

                variance_array.append(mean_var)
                r2_array.append(r2)
            sys.stdout = old_stdout
            variance_array.sort()
            r2_array.sort()

            to_render['bootstrap_confidence_99_lower'] = variance_array[0]
            to_render['bootstrap_confidence_99_upper'] = variance_array[1999]
            to_render['bootstrap_confidence_95_lower'] = variance_array[49]
            to_render['bootstrap_confidence_95_upper'] = variance_array[1949]
            to_render['bootstrap_confidence_90_lower'] = variance_array[99]
            to_render['bootstrap_confidence_90_upper'] = variance_array[1899]
            to_render['bootstrap_confidence_85_lower'] = variance_array[149]
            to_render['bootstrap_confidence_85_upper'] = variance_array[1849]
            to_render['bootstrap_confidence_80_lower'] = variance_array[199]
            to_render['bootstrap_confidence_80_upper'] = variance_array[1799]
            to_render['variance_array'] = variance_array

            my_model.bootstrap = to_render
            db.session.commit()
        return render_template('pages/models/bootstrap.html',
            to_render=to_render,
            my_data=my_data,
            variance=to_render['variance_array'] ,
            my_model=my_model)
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@model_blueprint.route('/predictions')
@login_required  # Limits access to authenticated users
def predictions_page():
    try:
        model_id = request.args.get('model_id')
        my_model = TrainedModel.query.filter_by(id=model_id).first()
        my_data = UserData.query.filter_by(id=my_model.data_id).first()
        if my_data.user_id is not current_user.id:
            flash('Opps!  Do data found', 'error')
            return redirect(request.referrer)
        predictions = Predictions.query.filter_by(model_id=my_model.id).all()

        return render_template('pages/models/predictions.html',
            my_data=my_data,
            predictions=predictions,
            my_model=my_model)
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@model_blueprint.route('/prediction')
@login_required  # Limits access to authenticated users
def prediction_page():
    try:
        dict_id = request.args.get('dict')
        my_dict = Predictions.query.filter_by(id=dict_id).first()
        my_model = TrainedModel.query.filter_by(id=my_dict.model_id).first()
        my_data = UserData.query.filter_by(id=my_model.data_id).first()
        if my_data.user_id is not current_user.id:
            flash('Opps!  Do data found', 'error')
            return redirect(request.referrer)

        examples = len(my_dict.predictions)
        orig_data = []
        apredictions = []

        for item in my_dict.predictions:
            apredictions.append(float(item))
        npredicted_data = np.array(apredictions)

        for x in range(0, len(my_dict.originals)):
            if my_dict.originals[x] is None or my_dict.originals[x] == "None":
                orig_data.append(None)
            else:
                orig_data.append(float(my_dict.originals[x]))
        norig_data = np.array(orig_data)

        to_render = {}
        results = {}
        variance = []
        scatter = []
        sorted_variance = []
        truth_table = {}
        san_vals = sanitize_results(norig_data, npredicted_data)
        norig_data = san_vals["orig"]
        npredicted_data = san_vals["pred"]

        if my_model.features['model_class'] == "predictor":
            for x in range(0, len(npredicted_data)):
                if norig_data[x] is None:
                    variance.append(np.float64(0.0))
                else:
                    variance.append(np.absolute(norig_data[x]-npredicted_data[x]))
                    scatter.append([norig_data[x],npredicted_data[x]])
            nvariance = np.array(variance)

            mean, sigma = np.mean(npredicted_data), np.std(npredicted_data)
            std_err = sigma / (sqrt(len(npredicted_data)))
            to_render['confidence_99'] = std_err * 2.575
            to_render['confidence_95'] = std_err * 1.96
            to_render['confidence_90'] = std_err * 1.645
            to_render['confidence_85'] = std_err * 1.44
            to_render['confidence_80'] = std_err * 1.28
            outlier = []
            upper = np.percentile(nvariance,75)
            lower = np.percentile(nvariance,25)
            for item in nvariance:
                if item > upper or item < lower:
                    outlier.append([0, item])
            explained, unexplained = compute_explained_variance(norig_data, npredicted_data)

            to_render['explained_variance'] = explained
            to_render['unexplained'] = unexplained
            to_render['outliers'] = outlier
            to_render['orig'] = {"min": round(np.nanmin(norig_data), 2), "max": round(np.nanmax(norig_data), 2), "mean": round(np.nanmean(norig_data), 2), "median": np.median(norig_data), "upper": np.percentile(norig_data,75), "lower": np.percentile(norig_data,25)}
            to_render['predicted'] = {"min": round(np.nanmin(npredicted_data), 2), "max": round(np.nanmax(npredicted_data), 2), "mean": round(np.nanmean(npredicted_data), 2), "median": np.median(npredicted_data), "upper": np.percentile(npredicted_data,75), "lower": np.percentile(npredicted_data,25)}
            to_render['variance'] = {"min": round(np.nanmin(nvariance), 2), "max": round(np.nanmax(nvariance), 2), "mean": round(np.nanmean(nvariance), 2), "median": np.median(nvariance), "upper": np.percentile(nvariance,75), "lower": np.percentile(nvariance,25)}
            to_render['scatter'] = scatter
            sorted_variance = sorted(variance)
        else:
            total_correct = 0
            total_missed = 0
            my_len = len(norig_data) - 1
            for x in range(0, my_len):
                if (str(norig_data[x]), str(npredicted_data[x])) not in truth_table:
                    truth_table[(str(norig_data[x]), str(npredicted_data[x]))] = 1
                else:
                    truth_table[(str(norig_data[x]), str(npredicted_data[x]))] = truth_table[(str(norig_data[x]), str(npredicted_data[x]))] + 1
                if norig_data[x] == npredicted_data[x]:
                    total_correct = total_correct + 1
                else:
                    total_missed = total_missed + 1
            to_render['accuracy'] = {"total_missed": total_missed, "total_correct": total_correct}
            try:
                results["f1_score"] = metrics.f1_score(norig_data, npredicted_data)
            except Exception as e:
                results["f1_score"] = None
            try:
                results["recall"] = metrics.recall_score(norig_data, npredicted_data)
            except Exception as e:
                results["recall"] = None
            try:
                results["precision"] = metrics.precision_score(norig_data, npredicted_data)
            except Exception as e:
                results["precision"] = None
            try:
                results["accuracy"] = metrics.accuracy_score(norig_data, npredicted_data)
            except Exception as e:
                results["accuracy"] = None

        return render_template('pages/models/prediction.html',
            my_data=my_data,
            my_dict=my_dict,
            my_model=my_model,
            col_name=my_model.features['target'],
            to_render=to_render,
            results=results,
            filename=os.path.basename(my_dict.oname),
            truth_table=truth_table,
            npredicted_data=npredicted_data,
            norig_data=norig_data,
            variance=variance,
            sorted_variance=sorted_variance,
            examples=examples)
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@model_blueprint.route('/predictions_step1', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def predictions_step1_page():
    # try:
        tc.config.set_num_gpus(0)
        model_id = request.args.get('model_id')
        my_model = TrainedModel.query.filter_by(id=model_id).first()
        my_data = UserData.query.filter_by(project_id=my_model.project_id).all()
        if my_data[0].user_id is not current_user.id:
            flash('Opps!  Do data found', 'error')
            return redirect(request.referrer)

        form = UserProfileForm(request.form, obj=current_user)
        if request.method == 'POST':
            data_id = request.form['data_set_id']
            my_data = UserData.query.filter_by(id=data_id).first()
            data_frame = tc.load_sframe(my_data.sname)
            if my_model.features['model_type'] == 'deep':
                tfrm = data_frame.to_dataframe()
                tfrm = tfrm.sort_values(by=[my_model.features["session_id"], my_model.features["time_field"]])
                data_frame = tc.SFrame(data=tfrm)
                data_frame[str(my_model.features["session_id"])] = data_frame[str(my_model.features["session_id"])].astype(int)

            model = tc.load_model(my_model.mname)
            predictions = model.predict(data_frame).to_numpy()

            my_dict = Predictions()
            my_dict.model_id = my_model.id
            my_dict.user_id = current_user.id
            my_dict.path = my_model.path
            my_dict.input_file = my_data.name
            my_predictions = []
            for item in predictions:
                my_predictions.append(str(item))
            my_dict.predictions = my_predictions
            origs = []
            for item in data_frame[str(my_model.features['target'])]:
                origs.append(str(item))

            # Make sure the predictions only overwrite blank values
            if request.form['mode'] == "fill":
                size = len(predictions)
                for x in range(0, size):
                    if origs[x] is not None:
                        predictions[x] = origs[x]

            my_dict.originals = origs
            data_frame = safely_add_col('Predicted_Value', predictions, data_frame)
            my_dict.oname = os.path.join(my_dict.path, str(uuid.uuid4())  + "_model_predictions.csv")
            data_frame.save(my_dict.oname, format='csv')
            db.session.add(my_dict)
            db.session.commit()

            # Redirect to home page
            return redirect(url_for('model.prediction_page', dict=my_dict.id))
        return render_template('pages/models/predict_step1.html',
            my_data=my_data,
            my_model=my_model,
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

@model_blueprint.route('/cross_validation')
@login_required  # Limits access to authenticated users
def cross_validation_page():
#    try:
        model_id = request.args.get('model_id')
        my_model = TrainedModel.query.filter_by(id=model_id).first()
        my_data = UserData.query.filter_by(id=my_model.data_id).first()
        if my_data.user_id is not current_user.id:
            flash('Opps!  Do data found', 'error')
            return redirect(request.referrer)

        feats = []
        imps = []
        data_frame = tc.load_sframe(my_data.sname)
        # data_frame = data_frame.sort(str(my_model.features['target']))
        df = shuffle(data_frame.to_dataframe())
        for y in range(0, 400):
            df = shuffle(df)
        data_frame = tc.SFrame(data=df)
        
        data_frame_cleaned = data_frame.dropna(str(my_model.features['target']), how="any")
        predicted_scores = []
        origs = []
        cols = []
        for feature in my_model.features['features']:
            data_frame_cleaned = data_frame_cleaned.dropna(str(feature), how="any")
            cols.append(str(feature))

        data_frame_cleaned = data_frame_cleaned.add_row_number(column_name='process_id')
        size = len(data_frame_cleaned)
        print(size)
        f = open(os.devnull, 'w')
        oldout = sys.stdout
        sys.stdout = f

        if my_model.cname is None:
            if my_model.features['model_class'] == "predictor":    
                predicted_scores = []  
                total_variance = 0                  
                for x in range(0, size):       
                    rowcomp = data_frame_cleaned[data_frame_cleaned['process_id'] == x]
                    row = rowcomp[0]
                    training_set = data_frame_cleaned[data_frame_cleaned['process_id'] != x]                                                 
                    if my_model.features['model_type'] == 'gradient':
                        tc_model = tc.boosted_trees_regression.create(training_set, target=str(my_model.features['target']), validation_set=None, features=cols, max_depth = my_model.options['max_depth'], max_iterations = my_model.options['max_iterations'] )
                    elif my_model.features['model_type'] == 'linear':
                        tc_model = tc.linear_regression.create(training_set, target=str(my_model.features['target']), validation_set=None, features=cols )
                    elif my_model.features['model_type'] == 'decision':
                        tc_model = tc.decision_tree_regression.create(training_set, target=str(my_model.features['target']), validation_set=None, features=cols, max_depth = my_model.options['max_depth'] )
                    elif my_model.features['model_type'] == 'random':
                        tc_model = tc.random_forest_regression.create(training_set, target=str(my_model.features['target']), validation_set=None, features=cols, max_depth = my_model.options['max_depth'], max_iterations = my_model.options['max_iterations'] )
                    predicted_scores.extend(tc_model.predict(row))
                    origs.append(row[str(my_model.features['target'])])
            else:
                for x in range(0, size):
                    rowcomp = data_frame_cleaned[data_frame_cleaned['process_id'] == x]
                    row = rowcomp[0]
                    training_set = data_frame_cleaned[data_frame_cleaned['process_id'] != x]
                    tc_model = None                             
                    if my_model.features['model_type'] == 'gradient':
                        tc_model = tc.boosted_trees_classifier.create(training_set, target=str(my_model.features['target']), validation_set=None, features=cols, max_depth = my_model.options['max_depth'], max_iterations = my_model.options['max_iterations'] )
                    elif my_model.features['model_type'] == 'linear':
                        tc_model = tc.logistic_classifier.create(training_set, target=str(my_model.features['target']), validation_set=None, features=cols )
                    elif my_model.features['model_type'] == 'decision':
                        tc_model = tc.decision_tree_classifier.create(training_set, target=str(my_model.features['target']), validation_set=None, features=cols, max_depth = my_model.options['max_depth'] )
                    elif my_model.features['model_type'] == 'random':
                        tc_model = tc.random_forest_classifier.create(training_set, target=str(my_model.features['target']), validation_set=None, features=cols, max_depth = my_model.options['max_depth'], max_iterations = my_model.options['max_iterations'] )
                    elif my_model.features['model_type'] == 'svm':
                        tc_model = tc.svm_classifier.create(training_set, target=str(my_model.features['target']), validation_set=None, features=cols, max_iterations = my_model.options['max_iterations'] )
                    predicted_scores.extend(tc_model.predict(row))
                    origs.append(row[str(my_model.features['target'])])
            
            residuals = []
            residual_percent = []               
            my_model.predictions = predicted_scores
            my_model.originals = origs
            size = len(predicted_scores)

            data_frame_cleaned = safely_add_col('Predicted_Value', predicted_scores, data_frame_cleaned)
            if my_model.features['model_class'] == "predictor":
                for j in range(0, size):
                    if  origs[j] is None or origs[j] == 0:
                        residuals.append(0)
                        residual_percent.append(0)
                    else:
                        res = abs(float(origs[j]) - float(predicted_scores[j]))
                        residuals.append(res)
                        residual_percent.append(res / float(origs[j]))
                data_frame_cleaned = safely_add_col('Residuals', residuals, data_frame_cleaned)
                data_frame_cleaned = safely_add_col('Percent Variance', residual_percent, data_frame_cleaned)
            my_model.cname = os.path.join(my_model.path, str(my_model.name) + str(my_model.api_key) + "_model_cross_validation.csv")
            data_frame_cleaned.save(my_model.cname, format='csv')

            db.session.commit()
        examples = len(my_model.predictions)

        orig_data = my_model.originals
        norig_data = np.array(orig_data)
        npredicted_data = np.array(my_model.predictions)

        to_render = {}
        variance = []
        sorted_variance = []
        scatter = []
        results = {}
        truth_table = {}
        if my_model.features['model_class'] == "predictor":
            for x in range(0, len(npredicted_data)):
                variance.append(np.absolute(float(norig_data[x])-float(npredicted_data[x])))
                scatter.append([float(norig_data[x]),float(npredicted_data[x])])
            nvariance = np.array(variance)
            mean, sigma = np.mean(npredicted_data), np.std(npredicted_data)
            std_err = sigma / (sqrt(len(npredicted_data)))
            to_render['confidence_99'] = std_err * 2.575
            to_render['confidence_95'] = std_err * 1.96
            to_render['confidence_90'] = std_err * 1.645
            to_render['confidence_85'] = std_err * 1.44
            to_render['confidence_80'] = std_err * 1.28
            outlier = []
            upper = np.percentile(nvariance,75)
            lower = np.percentile(nvariance,25)
            for item in nvariance:
                if item > upper or item < lower:
                    outlier.append([0, item])
            to_render['orig'] = {"min": round(np.nanmin(norig_data), 2), "max": round(np.nanmax(norig_data), 2), "mean": round(np.nanmean(norig_data), 2), "median": np.median(norig_data), "upper": np.percentile(norig_data,75), "lower": np.percentile(norig_data,25)}
            to_render['predicted'] = {"min": round(np.nanmin(npredicted_data), 2), "max": round(np.nanmax(npredicted_data), 2), "mean": round(np.nanmean(npredicted_data), 2), "median": np.median(npredicted_data), "upper": np.percentile(npredicted_data,75), "lower": np.percentile(npredicted_data,25)}
            to_render['variance'] = {"min": round(np.nanmin(nvariance), 2), "max": round(np.nanmax(nvariance), 2), "mean": round(np.nanmean(nvariance), 2), "median": np.median(nvariance), "upper": np.percentile(nvariance,75), "lower": np.percentile(nvariance,25)}
            to_render['scatter'] = scatter
            to_render['outliers'] = outlier
            explained, unexplained = compute_explained_variance(norig_data, npredicted_data)
            to_render['explained_variance'] = explained
            to_render['unexplained'] = unexplained
            sorted_variance = sorted(variance)
        else:
            total_correct = 0
            total_missed = 0
            my_len = len(norig_data)
            for x in range(0, my_len):
                if (str(norig_data[x]), str(npredicted_data[x])) not in truth_table:
                    truth_table[(str(norig_data[x]), str(npredicted_data[x]))] = 1
                else:
                    truth_table[(str(norig_data[x]), str(npredicted_data[x]))] = truth_table[(str(norig_data[x]), str(npredicted_data[x]))] + 1
                if norig_data[x] == npredicted_data[x]:
                    total_correct = total_correct + 1
                else:
                    total_missed = total_missed + 1
            try:
                to_render['accuracy'] = {"total_missed": total_missed, "total_correct": total_correct}
            except Exception as e:
                to_render['accuracy'] = None
            try:
                results["f1_score"] = metrics.f1_score(norig_data, npredicted_data)
            except Exception as e:
                results["f1_score"] = None
            try:
                results["recall"] = metrics.recall_score(norig_data, npredicted_data)
            except Exception as e:
                results["recall"] = None
            try:
                results["precision"] = metrics.precision_score(norig_data, npredicted_data)
            except Exception as e:
                results["precision"] = None
            try:
                results["accuracy"] = metrics.accuracy_score(norig_data, npredicted_data)
            except Exception as e:
                results["accuracy"] = None

        filename = str(my_model.name) + str(my_model.api_key) + '_model_cross_validation.csv'
        return render_template('pages/models/cross_validation.html',
            my_data=my_data,
            my_model=my_model,
            col_name=my_model.features['target'],
            to_render=to_render,
            filename=filename,
            results=results,
            truth_table=truth_table,
            npredicted_data=npredicted_data.tolist(),
            norig_data=norig_data.tolist(),
            variance=variance,
            sorted_variance=sorted_variance,
            examples=examples)
    # except Exception as e:
    #     flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
    #     error = ErrorLog()
    #     error.user_id = current_user.id
    #     error.error = str(e.__class__)
    #     error.parameters = request.args
    #     db.session.add(error)
    #     db.session.commit()
    #     return redirect(request.referrer)

@model_blueprint.route('/analytics')
@login_required  # Limits access to authenticated users
def model_analytics_page():
    try:
        model_id = request.args.get('model_id')
        my_model = TrainedModel.query.filter_by(id=model_id).first()
        my_data = UserData.query.filter_by(id=my_model.data_id).first()
        current_time = datetime.datetime.utcnow()
        thirty_days_ago = current_time - datetime.timedelta(days=30)
        runs = ModelRun.query.filter_by(model_id=my_model.id).filter(ModelRun.created_at>thirty_days_ago).limit(5000).all()
        values = []
        for run in runs:
            values.append(float(run.prediction))

        to_render = {}
        to_render['means'] = []
        to_render['min'] = []
        to_render['max'] = []
        to_render['std'] = []
        to_render['var'] = []
        to_render['runs_by_day'] = []
        narr = np.array([])
        for x in range(1, len(values)+1):
            narr = np.append(narr, values[x-1])
            to_render['means'].append(np.nanmean(narr))
            to_render['min'].append(np.nanmin(narr))
            to_render['max'].append(np.nanmax(narr))
            to_render['std'].append(np.nanstd(narr))
            to_render['var'].append(np.nanvar(narr))

        result = db.engine.execute("SELECT count(a.id) FROM (SELECT to_char(date_trunc('day', (current_date - offs)), 'YYYY-MM-DD') AS date FROM generate_series(0, 31, 1) AS offs) d LEFT OUTER JOIN runs a ON d.date = to_char(date_trunc('day', a.created_at), 'YYYY-MM-DD') AND a.model_id = '" + str(my_model.id) + "' GROUP BY d.date ORDER BY d.date")
        for item in result:
            to_render['runs_by_day'].append(int(item["count"]))

        return render_template('pages/models/analytics.html',
            to_render=to_render,
            my_data=my_data,
            my_model=my_model)
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@model_blueprint.route('/model_details')
@login_required  # Limits access to authenticated users
def model_details_page():
    # try:
        model_id = request.args.get('model_id')
        my_model = TrainedModel.query.filter_by(id=model_id).first()
        my_data = UserData.query.filter_by(id=my_model.data_id).first()
        if my_data.user_id is not current_user.id:
            flash('Opps!  Do data found', 'error')
            return redirect(request.referrer)
        feats = []
        imps = []
        optimal_lower = len(my_model.features["features"])*10 
        optimal_upper = len(my_model.features["features"])*20
        for item in my_model.features["importance"]:
            feats.append(str(item["name"]))
            imps.append(float(item["count"]))

        return render_template('pages/models/model_details.html',
            my_model=my_model,
            my_data=my_data,
            optimal_lower=optimal_lower,
            optimal_upper=optimal_upper,
            included_rows=(100-(float(my_model.features["training_loss"]))),
            missing_rows=float(my_model.features["training_loss"]),
            feats=feats,
            feat_len=len(my_model.features["features"]),
            imps=imps)
    # except Exception as e:
    #     flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
    #     error = ErrorLog()
    #     error.user_id = current_user.id
    #     error.error = str(e.__class__)
    #     error.parameters = request.args
    #     db.session.add(error)
    #     db.session.commit()
    #     return redirect(request.referrer)

@model_blueprint.route('/train_model', methods=['GET', 'POST'])
@login_required  # Limits access to authenticated users
def train_model_page():
    try:
        tc.config.set_num_gpus(0)
        data_id = request.args.get('data_id')
        my_data = UserData.query.filter_by(id=data_id).first()
        if my_data.user_id is not current_user.id:
            flash('Opps!  Do data found', 'error')
            return redirect(request.referrer)

        my_model = TrainedModel()
        form = TrainModelForm(request.form, obj=my_model)
        data_frame = tc.load_sframe(my_data.sname)

        if request.method == 'POST':
            form.populate_obj(my_model)
            model_type = request.form['model']
            model_class = request.form['model_class']
            max_depth = request.form['max_depth']
            max_iterations = request.form['max_iterations']
            session_id = str(request.form['session_id'])
            time_field = str(request.form['time_field'])
            my_model.mtype = model_type
            # data_frame = data_frame.sort(str(request.form['target']))
            if model_type == 'svm':
                label_count = data_frame[str(request.form['target'])].unique()
                if len(label_count) > 2:
                    flash('SVM only supports binary classification - try another method.', 'error')
                    return redirect(request.referrer)
            if model_type != 'deep':
                df = shuffle(data_frame.to_dataframe())
                for y in range(0, 500):
                    df = shuffle(df)
                data_frame = tc.SFrame(data=df)
            else:
                tfrm = data_frame.to_dataframe()
                tfrm = tfrm.sort_values(by=[session_id, time_field])
                data_frame = tc.SFrame(data=tfrm)
                data_frame[session_id] = data_frame[session_id].astype(int)

            options_dict = {}
            if max_depth is not None:
                options_dict['max_depth'] = int(max_depth)
            if max_iterations is not None:
                options_dict['max_iterations'] = int(max_iterations)
            data_frame_cleaned = data_frame.dropna(str(request.form['target']), how="any")
            cols = []
            for feature in request.form.getlist('features'):
                if str(feature) == str(request.form['target']):
                    flash('You can not select the target field in your training features.', 'error')
                    return redirect(url_for('model.train_model_page', data_id=data_id))
                data_frame_cleaned = data_frame_cleaned.dropna(str(feature), how="any")
                cols.append(str(feature))
            if data_frame_cleaned.num_rows() < 2:
                flash('After cleaning, there is no data left. You have a data quality issue.', 'error')
                return redirect(url_for('model.train_model_page', data_id=data_id))
            my_model.user_id = current_user.id
            print("USER ID")
            print(my_model.user_id)
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            training_loss = ((float(data_frame.num_rows()) - float(data_frame_cleaned.num_rows())) / float(data_frame.num_rows())) * 100

            train_data = None
            test_data = None
            if model_type != 'deep':
                train_data,test_data = data_frame_cleaned.random_split(.80,seed=0)
            else:
                train_data,test_data = tc.activity_classifier.util.random_split_by_session(data_frame_cleaned, session_id=session_id, fraction=0.8)

            tc_model = None
            # Setup options
            if model_type == 'gradient':
                if max_depth is None:
                    options_dict['max_depth'] = 6
                if max_iterations is None:
                    options_dict['max_iterations'] = 10
            elif model_type == 'linear':
                # Do nothing interesting
                options_dict = {}
            elif model_type == 'decision':
                if max_depth is None:
                    options_dict['max_depth'] = 6
            elif model_type == 'random':
                if max_depth is None:
                    options_dict['max_depth'] = 6
                if max_iterations is None:
                    options_dict['max_iterations'] = 10
            elif model_type == 'svm':
                if max_iterations is None:
                    options_dict['max_iterations'] = 10
            elif model_type == 'deep':
                if max_iterations is None:
                    options_dict['max_iterations'] = 10
            results = {}
            console = None
            if model_class == "predictor":
                best_run = None
                test_run = None
                working_results = {}
                working_train_data,working_test_data = data_frame_cleaned.random_split(.80,seed=0)
                num_rolls = 50
                if working_train_data.num_rows() < 1000:
                    num_rolls = 70
                if working_train_data.num_rows() < 500:
                    num_rolls = 100     
                if working_train_data.num_rows() < 200:
                    num_rolls = 200                                     
                for x in range(0, num_rolls):
                    working_train_data,working_test_data = data_frame_cleaned.random_split(.80)
                    if model_type == 'gradient':
                        test_run = tc.boosted_trees_regression.create(working_train_data, target=str(request.form['target']), validation_set=None, features=cols, max_depth = options_dict['max_depth'], max_iterations = options_dict['max_iterations'] )
                    elif model_type == 'linear':
                        test_run = tc.linear_regression.create(working_train_data, target=str(request.form['target']), validation_set=None, features=cols )
                    elif model_type == 'decision':
                        test_run = tc.decision_tree_regression.create(working_train_data, target=str(request.form['target']), validation_set=None, features=cols, max_depth = options_dict['max_depth'] )
                    elif model_type == 'random':
                        test_run = tc.random_forest_regression.create(working_train_data, target=str(request.form['target']), validation_set=None, features=cols, max_depth = options_dict['max_depth'], max_iterations = options_dict['max_iterations'] )
                    working_results = test_run.evaluate(working_test_data)
                    if best_run is None or working_results['max_error'] < best_run:
                        tc_model = test_run
                        train_data = working_train_data
                        test_data = working_test_data
                        results = working_results
                        best_run = results['max_error']
                        console = mystdout.getvalue()
                    mystdout.truncate(0)
            else:
                if model_type == 'gradient':
                    tc_model = tc.boosted_trees_classifier.create(train_data, target=str(request.form['target']), validation_set=None, features=cols, max_depth = options_dict['max_depth'], max_iterations = options_dict['max_iterations'])
                elif model_type == 'linear':
                    tc_model = tc.logistic_classifier.create(train_data, target=str(request.form['target']), validation_set=None, features=cols )
                elif model_type == 'decision':
                    tc_model = tc.decision_tree_classifier.create(train_data, target=str(request.form['target']), validation_set=None, features=cols, max_depth = options_dict['max_depth'] )
                elif model_type == 'random':
                    tc_model = tc.random_forest_classifier.create(train_data, target=str(request.form['target']), validation_set=None, features=cols, max_depth = options_dict['max_depth'], max_iterations = options_dict['max_iterations'] )
                elif model_type == 'svm':
                    tc_model = tc.svm_classifier.create(train_data, target=str(request.form['target']), validation_set=None, features=cols, max_iterations = options_dict['max_iterations'])
                elif model_type == 'deep':
                    tc_model = tc.activity_classifier.create(train_data, session_id=session_id, target=str(request.form['target']), validation_set=None, features=cols, max_iterations = options_dict['max_iterations'])
                results = tc_model.evaluate(test_data)
            my_model.user_id = current_user.id
            my_model.data_id = my_data.id
            my_model.project_id = my_data.project_id
            my_model.path = my_data.path
            my_model.options = options_dict
            my_model.api_key = str(uuid.uuid4())
            imp = []
            if model_type != 'linear' and model_type != 'svm' and model_type != 'deep':
                imp = tc_model.get_feature_importance()
            importance = []
            for col in imp:
                importance.append({"name": str(col["name"]), "index": str(col["index"]), "count": str(col["count"])})
            my_model.features = {"time_field": time_field, "session_id": session_id, "training_loss": training_loss, "training_rows": train_data.num_rows(), "test_rows": test_data.num_rows(), "features": cols, "target": request.form['target'], "importance": importance, "model_type": model_type, "model_class": model_class}

            sys.stdout = old_stdout
            if model_class == "predictor":
                my_model.results = results
                my_model.console = console
            else:
                print(results)
                if (model_type != 'svm'and model_type != 'deep'):
                    my_model.results = {'f1_score': nan_to_null(results['f1_score']), 'auc': nan_to_null(results['auc']), 'recall': nan_to_null(results['recall']), 'precision': nan_to_null(results['precision']), 'log_loss': nan_to_null(results['log_loss']), 'accuracy': nan_to_null(results['accuracy'])}
                else:
                    my_model.results = {'f1_score': nan_to_null(results['f1_score']), 'auc': "N/A", 'recall': nan_to_null(results['recall']), 'precision': nan_to_null(results['precision']), 'log_loss': "N/A", 'accuracy': nan_to_null(results['accuracy'])}
                my_model.console = mystdout.getvalue()
            my_model.mname = os.path.join(my_model.path, str(my_model.api_key) + "_model")
            tc_model.save(my_model.mname)  

            my_model.console = my_model.console.strip("\x00")
            db.session.add(my_model)
            db.session.commit()
            flash('Model has been saved!', 'success')
            return redirect(url_for('model.model_details_page', model_id=my_model.id))

        return render_template('pages/models/train_model_page.html',
            my_data=my_data,
            form=form,
            data_frame=data_frame,
            names=data_frame.column_names(),
            types=data_frame.column_types())
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP. Error: ' + str(e), 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)

@model_blueprint.route('/model_home')
@login_required  # Limits access to authenticated users
def model_home_page():
    try:
        my_models = TrainedModel.query.filter_by(user_id=current_user.id).all()
        return render_template('pages/models/model_page.html',
            my_models=my_models)
    except Exception as e:
        flash('Opps!  Something unexpected happened.  On the brightside, we logged the error and will absolutely look at it and work to correct it, ASAP.', 'error')
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return redirect(request.referrer)
