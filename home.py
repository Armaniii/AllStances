from flask import Flask, request,render_template, jsonify, redirect,url_for, Response
import main_v3 as m
from flask_scss import Scss
import requests
# from flask_sse import sse
from __init__ import create_app

#import waitress
#from waitress import serve
from pathlib import Path
from flask import Flask, flash
from flask_login import LoginManager,login_user, logout_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from database import User
from werkzeug.security import generate_password_hash, check_password_hash
from flask_bcrypt import Bcrypt

app = create_app()#.app_context().push()
args = {}

app.secret_key='Secret'

@app.route("/", methods=["GET", "POST"])
def home():
    arguments = {}
    # if user is not logged in and tries to access home page, redirect to login page
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    # if user is logged in and tries to access home page, render home page
    if request.method == "GET":
        return render_template("index.html",arguments = arguments, user = current_user)
    
    if request.method == "POST":
        use_reddit = bool(request.form.get("use-reddit", False))
        use_congress = bool(request.form.get("use-congress", False))
        use_general = bool(request.form.get("use-general", False))
        
        print("use reddit: ", bool(use_reddit))
        print("use congress: ", bool(use_congress))
        print("use general: ", bool(use_general))


        if use_reddit == False and use_congress == False and use_general == False:
            flash("Please select at least one source of information", category="error")
            print("Please select at least one source of information")
            return redirect(url_for('home'))
        process_list(request.form)

    
    return render_template("index.html",arguments = arguments, user = current_user)

@app.route("/publish", methods=["GET","POST"])
@login_required
def publish():
    return args


def process_list(form_data):
    topic = form_data["search_query"]
    use_reddit = form_data.get("use-reddit", False)
    use_congress = form_data.get("use-congress", False)
    use_general = form_data.get("use-general", False)

    slider_value = form_data.get('slider', "0.5")
    slider_value = float(slider_value)
    print("use reddit: ", use_reddit)
    print("use congress: ", use_congress)
    print("Diversity = " + str(slider_value))
    global args
    args = m.query(topic,use_reddit,use_congress,slider_value,use_general)
    print("finished getting arguments")

@app.route("/login", methods=["GET", "POST"])
def login():
    with app.app_context():
        if request.method == "GET":
            return render_template("login.html",user=current_user)
        if request.method == "POST":
            print(app)
            bcrypt = Bcrypt(app)
            print("THIS IS THE APP CONTEXT")
            username = request.form["text"]
            password = request.form["password"]
            user = User.query.filter_by(username=username).first()
            if not user or not bcrypt.check_password_hash(user.password, password):
                flash("Invalid Username or Password..", category="error")
                print("Please check your login details and try again.")
                return redirect(url_for('login'))
            else:
                login_user(user,remember=True)
                return redirect(url_for('home'))
        return render_template("login.html",user=current_user)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

#if __name__ == "__main__":
      # waitress.serve(app, listen='127.0.0.1:5000')
#     app.run(host='0.0.0.0',port=8080,debug=True)
#     # serve(app)
