import os

# *****************************
# Environment specific settings
# *****************************

# DO NOT use "DEBUG = True" in production environments
DEBUG = True

# DO NOT use Unsecure Secrets in production environments
# Generate a safe one with:
#     python -c "import os; print repr(os.urandom(24));"
SECRET_KEY = 'This is an UNSECURE Secret. CHANGE THIS for production environments.'

# SQLAlchemy settings
SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:welcome@localhost/iomodel'
#SQLALCHEMY_DATABASE_URI = 'sqlite:///../app.sqlite'
SQLALCHEMY_TRACK_MODIFICATIONS = False    # Avoids a SQLAlchemy Warning

# Flask-Mail settings
# For smtp.gmail.com to work, you MUST set "Allow less secure apps" to ON in Google Accounts.
# Change it in https://myaccount.google.com/security#connectedapps (near the bottom).
MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 587
MAIL_USE_SSL = False
MAIL_USE_TLS = True
MAIL_USERNAME = 'you@gmail.com'
MAIL_PASSWORD = 'yourpassword'
MAIL_DEFAULT_SENDER = '"You" <you@gmail.com>'

ALLOWED_EXTENSIONS = set(['csv'])
UPLOAD_FOLDER = '/home/matt/projects/iomodel/uploads'
APP_FOLDER = '/home/matt/projects/iomodel'

ADMINS = [
    '"Admin One" <admin1@gmail.com>',
    ]
