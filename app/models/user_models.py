# Copyright 2014 SolidBuilds.com. All rights reserved
#
# Authors: Ling Thio <ling.thio@gmail.com>, Matt Hogan <matt@twintechlabs.io>

from flask_user import UserMixin
from flask_user.forms import RegisterForm
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, validators
from app import db
import datetime
from sqlalchemy import Column, Integer, DateTime

# Define the User data model. Make sure to add the flask_user.UserMixin !!
class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)

    # User authentication information (required for Flask-User)
    email = db.Column(db.Unicode(255), nullable=False, server_default=u'', unique=True)
    email_confirmed_at = db.Column(db.DateTime())
    password = db.Column(db.String(255), nullable=False, server_default='')
    # reset_password_token = db.Column(db.String(100), nullable=False, server_default='')
    active = db.Column(db.Boolean(), nullable=False, server_default='0')

    # User information
    active = db.Column('is_active', db.Boolean(), nullable=False, server_default='0')
    first_name = db.Column(db.Unicode(50), nullable=False, server_default=u'')
    last_name = db.Column(db.Unicode(50), nullable=False, server_default=u'')

    # Relationships
    roles = db.relationship('Role', secondary='users_roles',
                            backref=db.backref('users', lazy='dynamic'))
    def has_role(self, role):
        for item in self.roles:
            if item.name == 'admin':
                return True
        return False

    def role(self):
        print(self.roles)
        for item in self.roles:
            return item.name

    def name(self):
        return self.first_name + " " + self.last_name


# Define the Role data model
class Role(db.Model):
    __tablename__ = 'roles'
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(50), nullable=False, server_default=u'', unique=True)  # for @roles_accepted()
    label = db.Column(db.Unicode(255), server_default=u'')  # for display purposes


# Define the UserRoles association model
class UsersRoles(db.Model):
    __tablename__ = 'users_roles'
    id = db.Column(db.Integer(), primary_key=True)
    user_id = db.Column(db.Integer(), db.ForeignKey('users.id', ondelete='CASCADE'))
    role_id = db.Column(db.Integer(), db.ForeignKey('roles.id', ondelete='CASCADE'))


# Define the User registration form
# It augments the Flask-User RegisterForm with additional fields
class MyRegisterForm(RegisterForm):
    first_name = StringField('First name', validators=[
        validators.DataRequired('First name is required')])
    last_name = StringField('Last name', validators=[
        validators.DataRequired('Last name is required')])


# Define the User profile form
class UserProfileForm(FlaskForm):
    first_name = StringField('First name', validators=[
        validators.DataRequired('First name is required')])
    last_name = StringField('Last name', validators=[
        validators.DataRequired('Last name is required')])
    submit = SubmitField('Save')

class StorageSlice(db.Model):
    __tablename__ = 'slices'
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(50), nullable=False, server_default=u'', unique=True)  # for @roles_accepted()
    
class Project(db.Model):
    __tablename__ = 'projects'
    id = db.Column(db.Integer(), primary_key=True)
    user_id = db.Column(db.Integer(), db.ForeignKey('users.id', ondelete='CASCADE'))
    name = db.Column(db.Unicode(255), nullable=False, server_default=u'', unique=False)
    description = db.Column(db.Unicode(2055), nullable=False, server_default=u'', unique=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def num_sets(self):
        return UserData.query.filter_by(project_id=self.id).count()

    def num_models(self):
        return TrainedModel.query.filter_by(project_id=self.id).count()

class UserData(db.Model):
    __tablename__ = 'users_data'
    id = db.Column(db.Integer(), primary_key=True)
    user_id = db.Column(db.Integer(), db.ForeignKey('users.id', ondelete='CASCADE'))
    project_id = db.Column(db.Integer(), db.ForeignKey('projects.id', ondelete='CASCADE'))
    name = db.Column(db.Unicode(255), nullable=False, server_default=u'', unique=False)
    description = db.Column(db.Unicode(2055), nullable=False, server_default=u'', unique=False)
    path = db.Column(db.Unicode(255), nullable=False, server_default=u'', unique=False)
    fname = db.Column(db.Unicode(255), nullable=False, server_default=u'', unique=False)
    sname = db.Column(db.Unicode(255), nullable=False, server_default=u'', unique=False)
    num_rows = db.Column(db.Integer(), nullable=False, unique=False)
    num_cols = db.Column(db.Integer(), nullable=False, unique=False)
    stats = db.Column(db.JSON)
    uploaded_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    project = db.relationship(
        Project,
        backref=db.backref('users_data',
                         uselist=True,
                         cascade='delete,all'))

class DBConn(db.Model):
    __tablename__ = 'conns'
    id = db.Column(db.Integer(), primary_key=True)
    user_id = db.Column(db.Integer(), db.ForeignKey('users.id', ondelete='CASCADE'))
    name = db.Column(db.String(255), server_default='')
    engine_type = db.Column(db.String(255), server_default='')
    user = db.Column(db.String(255), server_default='')
    password = db.Column(db.String(255), server_default='')
    host = db.Column(db.String(255), server_default='')
    db = db.Column(db.String(255), server_default='')

class TrainedModel(db.Model):
    __tablename__ = 'models'
    id = db.Column(db.Integer(), primary_key=True)
    user_id = db.Column(db.Integer(), db.ForeignKey('users.id', ondelete='CASCADE'))
    data_id = db.Column(db.Integer(), db.ForeignKey('users_data.id', ondelete='CASCADE'))
    project_id = db.Column(db.Integer(), db.ForeignKey('projects.id', ondelete='CASCADE'))
    name = db.Column(db.String(255), nullable=False, server_default=u'', unique=False)
    mtype = db.Column(db.String(30), nullable=False, server_default=u'', unique=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    path = db.Column(db.String(255), nullable=False, server_default=u'', unique=False)
    mname = db.Column(db.String(255), nullable=False, server_default=u'', unique=False)
    cname = db.Column(db.String(255), nullable=True, unique=False)
    api_key = db.Column(db.String(80), nullable=False, server_default=u'')
    results = db.Column(db.JSON)
    bootstrap = db.Column(db.JSON)
    features = db.Column(db.JSON)
    options = db.Column(db.JSON)
    predictions = db.Column(db.JSON)
    originals = db.Column(db.JSON)
    console = db.Column(db.Text, server_default=u'')
    project = db.relationship(
        Project,
        backref=db.backref('models',
                         uselist=True,
                         cascade='delete,all'))
    data = db.relationship(
        UserData,
        backref=db.backref('models',
                         uselist=True,
                         cascade='delete,all'))
    
    def model_status(self):
        if self.results['auc'] == None:
            return "failed"
        else:
            return "valid"

class ClusterAnalysis(db.Model):
    __tablename__ = 'cluster_analysis'
    id = db.Column(db.Integer(), primary_key=True)
    user_id = db.Column(db.Integer(), db.ForeignKey('users.id', ondelete='CASCADE'))
    data_id = db.Column(db.Integer(), db.ForeignKey('users_data.id', ondelete='CASCADE'))
    derived_data_id = db.Column(db.Integer())
    project_id = db.Column(db.Integer(), db.ForeignKey('projects.id', ondelete='CASCADE'))
    name = db.Column(db.Unicode(255), nullable=False, server_default=u'', unique=False)
    cluster_type = db.Column(db.Unicode(30), nullable=False, server_default=u'', unique=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    params = db.Column(db.JSON)
    project = db.relationship(
        Project,
        backref=db.backref('cluster_analysis',
                         uselist=True,
                         cascade='delete,all'))
    data = db.relationship(
        UserData,
        backref=db.backref('cluster_analysis',
                         uselist=True,
                         cascade='delete,all'))   

class Predictions(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer(), primary_key=True)
    user_id = db.Column(db.Integer(), db.ForeignKey('users.id', ondelete='CASCADE'))
    model_id = db.Column(db.Integer(), db.ForeignKey('models.id', ondelete='CASCADE'))
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    path = db.Column(db.Unicode(255), nullable=False, server_default=u'', unique=False)
    oname = db.Column(db.Unicode(255), nullable=True, unique=False)
    input_file = db.Column(db.Unicode(255), nullable=True, unique=False)
    predictions = db.Column(db.JSON)
    originals = db.Column(db.JSON)

class ModelRun(db.Model):
    __tablename__ = 'runs'
    id = db.Column(db.Integer(), primary_key=True)
    model_id = db.Column(db.Integer(), db.ForeignKey('models.id', ondelete='CASCADE'))
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    prediction = db.Column(db.String(50), nullable=True, unique=False)
    parameters = db.Column(db.JSON)

class ErrorLog(db.Model):
    __tablename__ = 'log'
    id = db.Column(db.Integer(), primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    error = db.Column(db.String(50), nullable=True, unique=False)
    user_id = db.Column(db.Integer(), db.ForeignKey('users.id', ondelete='CASCADE'))
    parameters = db.Column(db.JSON)

class MyRegisterForm(RegisterForm):
    first_name = StringField('First name', validators=[
        validators.DataRequired('First name is required')])
    last_name = StringField('Last name', validators=[
        validators.DataRequired('Last name is required')])

class UserProfileForm(FlaskForm):
    first_name = StringField('First name', validators=[
        validators.DataRequired('First name is required')])
    last_name = StringField('Last name', validators=[
        validators.DataRequired('Last name is required')])
    submit = SubmitField('Save')

class UserDataForm(FlaskForm):
    name = StringField('name', validators=[])
    description = StringField('description', validators=[])
    submit = SubmitField('Save')

class TrainModelForm(FlaskForm):
    name = StringField('name', validators=[])
    submit = SubmitField('Train')
