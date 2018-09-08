#
# Copyright 2018 Twin Tech Labs
# Author: Matt Hogan
#
from flask import Blueprint, redirect, render_template
from flask import request, url_for, flash, send_from_directory, jsonify, render_template_string
from flask_user import current_user, login_required, roles_accepted
from werkzeug.utils import secure_filename
import turicreate as tc
import sys
import numpy as np
from scipy import stats as scipy_stats

from app import db
from app.models.user_models import  TrainedModel, ModelRun, ErrorLog
import uuid, json, os
import datetime

_local_cache = {}

api_blueprint = Blueprint('api', __name__, template_folder='templates')

@api_blueprint.route('/predict', methods=['POST'])
def predict_page():
    try:
        api_key = request.args.get('api_key')
        model = None
        
        if api_key in _local_cache:
            model = _local_cache[api_key]
        else:
            my_model = TrainedModel.query.filter_by(api_key=api_key).first() 
            tc_model = tc.load_model(my_model.mname)  
            _local_cache[api_key] = { 'model_id': my_model.id, 'tc_model': tc_model }
            model = { 'model_id': my_model.id, 'tc_model': tc_model }
        sf = tc.SFrame([request.json])
        sf = sf.unpack('X1', column_name_prefix='')
        predicted_scores = model['tc_model'].predict(sf)
        run = ModelRun()
        run.model_id = model['model_id']
        run.parameters = request.json
        run.prediction = str(predicted_scores[0])
        db.session.add(run)
        db.session.commit()

        ret = {"prediction": round(predicted_scores[0])}
        return(jsonify(ret), 200)  
    except Exception as e:
        error = ErrorLog()
        error.user_id = current_user.id
        error.error = str(e.__class__)
        error.parameters = request.args
        db.session.add(error)
        db.session.commit()
        return(jsonify({"error": "That was unexpected. We logged it and are looking into it."}), 500)  
        
@api_blueprint.route('/predict_moca', methods=['POST'])
def predict_moca_page():
    model = None
    api_key = "7de9e4a7-d9b1-44dc-8531-2fbc3c20f272"
    if api_key in _local_cache:
        model = _local_cache[api_key]
    else:
        my_model = TrainedModel.query.filter_by(api_key=api_key).first() 
        tc_model = tc.load_model(my_model.mname)  
        _local_cache[api_key] = { 'model_id': my_model.id, 'tc_model': tc_model }
        model = { 'model_id': my_model.id, 'tc_model': tc_model }

    sf = tc.SFrame([request.json])
    sf = sf.unpack('X1', column_name_prefix='')
    predicted_scores = model['tc_model'].predict(sf)
    run = ModelRun()
    run.model_id = model['model_id']
    run.parameters = request.json
    run.prediction = str(predicted_scores[0])
    db.session.add(run)
    db.session.commit()

    classification = "Impaired (8-27)"
    if predicted_scores[0] >= 27:
        classification = "Within normal limits"
    elif predicted_scores[0] < 27 and predicted_scores[0] >= 26:
        classification = "Borderline"
    elif predicted_scores[0] < 26 and predicted_scores[0] >= 22:
        classification = "Mild Impairment"
    elif predicted_scores[0] < 22:
        classification = "Impaired"

    lower_bound = round(predicted_scores[0]-1.17)
    upper_bound = round(predicted_scores[0]+2.61)
    if predicted_scores[0] >= 28:
        lower_bound=28
    if predicted_scores[0] >= 27 and predicted_scores[0] < 28:
        lower_bound = 27    

    if upper_bound > 30:   
        upper_bound = 30  

    ret = {"predicted_moca": round(predicted_scores[0]), "classification": classification, "lower_bound": lower_bound, "upper_bound": upper_bound}
    return(jsonify(ret), 200)    

@api_blueprint.route('/predict_moca_eas', methods=['POST'])
def predict_eas_page():
    model = None
    api_key = "8d36535f-861f-498a-ae9a-2a027ba5eed6"
    if api_key in _local_cache:
        model = _local_cache[api_key]
    else:
        my_model = TrainedModel.query.filter_by(api_key=api_key).first() 
        tc_model = tc.load_model(my_model.mname)  
        _local_cache[api_key] = { 'model_id': my_model.id, 'tc_model': tc_model }
        model = { 'model_id': my_model.id, 'tc_model': tc_model }

    sf = tc.SFrame([request.json])
    sf = sf.unpack('X1', column_name_prefix='')
    predicted_scores = model['tc_model'].predict(sf)
    run = ModelRun()
    run.model_id = model['model_id']
    run.parameters = request.json
    run.prediction = str(predicted_scores[0])
    db.session.add(run)
    db.session.commit()

    classification = "Impaired (8-27)"
    if predicted_scores[0] >= 27:
        classification = "Within normal limits"
    elif predicted_scores[0] < 27 and predicted_scores[0] >= 26:
        classification = "Borderline"
    elif predicted_scores[0] < 26 and predicted_scores[0] >= 22:
        classification = "Mild Impairment"
    elif predicted_scores[0] < 22:
        classification = "Impaired"

    lower_bound = round(predicted_scores[0]-2.07)
    upper_bound = round(predicted_scores[0]+2.10)
    if predicted_scores[0] >= 28:
        lower_bound=28
    if predicted_scores[0] >= 27 and predicted_scores[0] < 28:
        lower_bound = 27    

    if upper_bound > 30:   
        upper_bound = 30        

    ret = {"predicted_moca": round(predicted_scores[0]), "classification": classification, "lower_bound": lower_bound, "upper_bound": upper_bound}
    return(jsonify(ret), 200)  