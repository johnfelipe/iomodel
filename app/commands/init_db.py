#
# Copyright 2018 Twin Tech Labs
# Author: Matt Hogan
#
import datetime

from flask import current_app
from flask_script import Command

from app import db
from app.models.user_models import User, Role, StorageSlice

class InitDbCommand(Command):
    """ Initialize the database."""

    def run(self):
        init_db()

def init_db():
    """ Initialize the database."""
    db.drop_all()
    db.create_all()
    create_users()


def create_users():
    """ Create users """

    # Create all tables
    db.create_all()

    # Adding roles
    admin_role = find_or_create_role('admin', u'Admin')

    # Add users
    user = find_or_create_user(u'Admin', u'User', u'admin@example.com', 'Password1', admin_role)
    user = find_or_create_user(u'Member', u'Example', u'member@example.com', 'Password1')

    sslice = find_or_create_slice(u'slice1')
    sslice = find_or_create_slice(u'slice2')

    # Save to DB
    db.session.commit()


def find_or_create_role(name, label):
    """ Find existing role or create new role """
    role = Role.query.filter(Role.name == name).first()
    if not role:
        role = Role(name=name, label=label)
        db.session.add(role)
    return role


def find_or_create_user(first_name, last_name, email, password, role=None):
    """ Find existing user or create new user """
    user = User.query.filter(User.email == email).first()
    if not user:
        user = User(email=email,
                    first_name=first_name,
                    last_name=last_name,
                    password=current_app.user_manager.hash_password(password),
                    active=True,
                    email_confirmed_at=datetime.datetime.utcnow())
        if role:
            user.roles.append(role)
        db.session.add(user)
    return user

def find_or_create_slice(name):
    """ Find existing user or create new user """
    sslice = StorageSlice.query.filter(StorageSlice.name == name).first()
    if not sslice:
        sslice = StorageSlice(name=name)

        db.session.add(sslice)
    return sslice

