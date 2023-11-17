
from pathlib import Path
from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_scss import Scss
from flask_sse import sse
import os 
db = SQLAlchemy()
database_name = "database.db"

def create_app():

        app = Flask(__name__,template_folder='/home/arman/allsides/web/templates',static_folder='/home/arman/allsides/web/static')
        #app.config["REDIS_URL"] = "redis://localhost:6379/0" #"redis://localhost:8080/0" #
        #app.register_blueprint(sse,url_prefix='/stream')
        app.config['SECRET_KEY'] = 'Secret'
        Scss(app, static_dir='/home/arman/allsides/web/static', asset_dir='/home/arman/allsides/web/assets')
        app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{database_name}'
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        with app.app_context():
            db.init_app(app)

            create_database_if_not_exist(app)
            from database import User
            
            

            login_manager = LoginManager()
            login_manager.login_view = 'login'
            login_manager.init_app(app)

            @login_manager.user_loader
            def load_user(id):
                return User.query.get(id)   

            return app
def create_database_if_not_exist(app):
        if not Path('/home/arman/allsides/web/website/' + database_name).exists():
            db.create_all()
            print('Created Database!')

