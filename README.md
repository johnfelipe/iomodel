# ioModel Research Platform - v1.0.0

![Screenshot](https://github.com/twintechlabs/iomodel/blob/master/app/static/images/screenshot.png)

This is the core code base of the ioModel Research Platform. ioModel provides a UI and
an intuitive workflow on top of a number of common open source machine learning libraries including
Apple's Turi Create, NumPy, ScyPi, and Scikit Learn.

Today - ioModel allows you to import your data via CSV and DB extract, analyze and transform the data, train machine
learning models (classifiers and predictors currently supported), evaluate their performance, and 
manage them as a deployed RESTful endpoint for integration into other systems.

Read more about it in the primer located here:
http://twintechlabs.io/Twin%20Tech%20Labs%20Primer.pdf

## Disclaimer

ioModel is under active development, is in the early stages of release, and may contain bugs. We'll fix them as soon as they come up. Software is provided as-is.

## Code characteristics

* Tested on Python 3.6 and 2.7
* Well organized directories with lots of comments
    * app
        * commands
        * models
        * static
        * templates
        * views
    * tests
* Includes test framework (`py.test` and `tox`)
* Includes database migration framework (`alembic`)
* Sends error emails to admins for unhandled exceptions


## Setting up a development environment

We assume that you have `git` and `virtualenv` and `virtualenvwrapper` installed.

    # Clone the code repository into ~/dev/my_app
    mkdir -p ~/dev
    cd ~/dev
    git clone https://github.com/twintechlabs/iomodel.git iomodel

    pip install -r requirements.txt


# Configuring SMTP

Edit the `local_settings.py` file.

Specifically set all the MAIL_... settings to match your SMTP settings

Note that Google's SMTP server requires the configuration of "less secure apps".
See https://support.google.com/accounts/answer/6010255?hl=en

Note that Yahoo's SMTP server requires the configuration of "Allow apps that use less secure sign in".
See https://help.yahoo.com/kb/SLN27791.html


## Initializing the Database

Create a database for your app and then run:

    # Create DB tables and populate the roles and users tables
    python manage.py init_db

    # Or if you have Fabric installed:
    fab init_db

NOTE: There is currently a bug on Ubuntu 18.04 where you may see this error:

    python3: Relink `/lib/x86_64-linux-gnu/libudev.so.1' with `/lib/x86_64-linux-gnu/librt.so.1' for IFUNC symbol `clock_gettime'
    Segmentation fault

If you do, run the following and then reinitialize the database:
    sudo apt-get install python3-opencv

## Setting up paths for local data frame storage

ioModel uses a database for managing users and relationships between models and data sets as well as for caching computationally expensive operations on immutable data. However, the file system is used to store raw data frames and a number of intermediary file types.

This section needs to be expanded to explain how to use the director scheme with network file mounts to support growable, sustainable storage, however, to get things running for now:

    # In the ioModel app directory you cloned from GitHUb:
    mkdir uploads
    cd uploads
    mkdir slice1
    mkdir slice2

    # Then, edit app/local_settings.py to point the following two variables to appropriate paths for your install:
    UPLOAD_FOLDER = '/home/YOUR_USER/iomodel/uploads'
    APP_FOLDER = '/home/YOUR_USER/iomodel/'

## Running the app

    # Start the Flask development web server
    python manage.py runserver

    # Or if you have Fabric installed:
    fab runserver

Point your web browser to http://localhost:5000/

You can make use of the following users:
- email `user@example.com` with password `Password1`.
- email `admin@example.com` with password `Password1`.


## Running the automated tests

    # Start the Flask development web server
    py.test tests/

    # Or if you have Fabric installed:
    fab test


## Trouble shooting

If you make changes in the Models and run into DB schema issues, delete the sqlite DB file `app.sqlite`.


## Acknowledgements

With thanks to the following libraries - we all build on the shoulder's of giants:
* [Apple's Turi Create](https://github.com/apple/turicreate)
* [NumPy](http://www.numpy.org/)
* [ScyPi](https://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [Pandas](https://pandas.pydata.org/)
* [FlaskDash](https://github.com/twintechlabs/flaskdash)
* [CoreUI](https://coreui.io/)
* [Alembic](http://alembic.zzzcomputing.com/)
* [Flask](http://flask.pocoo.org/)
* [Flask-Login](https://flask-login.readthedocs.io/)
* [Flask-Migrate](https://flask-migrate.readthedocs.io/)
* [Flask-Script](https://flask-script.readthedocs.io/)
* [Flask-User](http://flask-user.readthedocs.io/)

## Authors
- Matt Hogan - matt AT twintechlabs DOT io