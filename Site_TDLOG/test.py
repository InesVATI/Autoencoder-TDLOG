from flask import Flask, url_for, redirect, render_template, request
import numpy as np
import potentials as pt
import dihedral_angles as rama
import sqlite3
import click
from flask import Flask, url_for, abort, redirect, render_template, request, current_app, g, flash
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
import email_validator
from flask_user import UserManager
from flask_bcrypt import Bcrypt
from flask.cli import with_appcontext
import plotly
import os

N=1000

app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))
print(basedir)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'app.sqlite')


db= SQLAlchemy(app)
bcrypt=Bcrypt(app)

app.config['SECRET_KEY'] = 'thisisasecretkey'
#app.config['ALCHEMY_DATABASE_URI']='sqlite:///database.db'
#app.config['ALCHEMY_DATABASE_URI']= os.getenv('DATABASE_URL', 'sqlite:///app.db')
app.config['SQLACLHEMY_TRACK_MODIFICATIONS']=True

def fetch_url(name):
     url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"+name+"/SDF?record_type=3d"
     return url

class User(db.Model,UserMixin):
    id=db.Column(db.Integer, primary_key=True)
    username=db.Column(db.String(20),nullable=False, unique=True)
    password=db.Column(db.String(80),nullable=False)
    tuto0=db.Column(db.Boolean, nullable=True)
    tuto1=db.Column(db.Boolean, nullable=True)
    tuto2=db.Column(db.Boolean, nullable=True)
    tuto3=db.Column(db.Boolean, nullable=True)
    tuto4=db.Column(db.Boolean, nullable=True)

    def change_tuto(self,nbr):
        if nbr==0:
            self.tuto0=True
        if nbr==1:
            self.tuto1=True

class RegisterForm(FlaskForm):
    username=StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder":'username'})
    password=PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder":'Password'})
    submit=SubmitField("Register")

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError("That username already exists. Please choose a different one.")


class LoginForm(FlaskForm):
    username=StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder":'username'})
    password=PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder":'Password'})
    submit=SubmitField("Login")


login_manager=LoginManager()
login_manager.init_app(app)
#user_manager = UserManager(app, db, User)
login_manager.login_view="login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


#***********************************************************************************************************

@app.route('/register', methods=['GET', 'POST'])
def register():
    
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(
            form.password.data).decode('utf-8')
        new_user = User(username=form.username.data,
                        password=hashed_password, tuto0=False, tuto1=False, tuto2=False, tuto3=False, tuto4=False)
        db.session.add(new_user)
        db.session.commit()
        flash("Your account has successfully been created!")
        return redirect(url_for('login')) 
    return render_template('register.html', form = form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    
    form = LoginForm()
    if form.validate_on_submit():
        user=User.query.filter_by(username=form.username.data).first()
        if user :
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                flash("Login Succesfull")
                return redirect(url_for('index'))
            else : 
                flash("Wrong password - Try Again !")
        else :
            flash("That user doesn't exist - Try Again !")
    
    return render_template('login.html', form = form)

@app.route('/logout', methods=['GET','POST'])
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/', methods=['GET','POST'])
#@app.route('/index')
def index():
    return render_template('index.html', title='Homepage')


@app.route('/visualisation',  methods=['GET','POST'])
def visualisation():
    if request.method == "POST":
        molecule = request.form['molecule']
        beta = int(request.form['beta'])
        num = int(request.form['number'])
        print(num)
        bowls=[]
        
        for i in range(1,num+1):
            coord="bowl"+str(i)+"xy"
            xy=request.form[coord]
            x0,y0=xy.split(";")
            x0=float(x0)
            y0=float(y0)

            r_id="bowl"+str(i)+"r"
            r=float(request.form[r_id])
            a_id="bowl"+str(i)+"a"
            a=float(request.form[a_id])
            bowls.append([x0,y0,r,a])
        
        bowl=np.array(bowls)
        url = fetch_url(molecule)
        potential=pt.MultimodalPotential(bowl, beta)
        fig=pt.create_plots(potential)
        print(os.path.join(basedir, 'static/plot/plot3D.html'))
        fig.write_html(os.path.join(basedir, 'static/plot/plot3D.html'))
        fig=pt.plot_trajectory(potential) 
        fig.savefig(os.path.join(basedir, 'static/img/test.png'))
        choix_mod={"molecule": molecule , "molecule3D" : url  }
        return render_template('visualisation.html', title='Visualisation', choix=choix_mod , url= fetch_url(molecule), formulaire_rempli = True)

        
    return render_template('visualisation.html', title='Visualisation', formulaire_rempli=False)

@app.route('/explication/0', methods=['GET','POST'])
def explication():
    """ Vizualisation of our work during MOPSI project """
    current_user.tuto0=True
    return render_template('explication.html', title='Explanation')


@app.route('/profil/<string:username>/<string:status>')
def profil_visualization_history(username,status):
    return render_template('profil_visualization_history.html', username=username, status=status, title="Visualization history")

@app.route('/profil/<string:username>/<string:status>/parameters')
def profil_change_parameters(username,status):
    return render_template('profil_change_parameters.html', username=username, status=status, title="Change parameters")

@app.route('/profil/<string:username>/<string:status>/tutorial')
def profil_tutorial(username,status):
    return render_template('profil_tutorial.html', username=username, status=status, title="Tutorial")

@app.route('/explication/1')
def explication1():
    """ Vizualisation of our work during MOPSI project """

    current_user.tuto1=True
    if current_user.tuto1:
        print("L'uilisateur a fait le tuto1")
    return render_template('explication1.html', title='Explanation')

@app.route('/explication/2')
def explication2():
    """ Vizualisation of our work during MOPSI project """

    current_user.tuto2=True
    if current_user.tuto2:
        print("L'uilisateur a fait le tuto2")
    return render_template('explication2.html', title='Explanation')

@app.route('/explication/3')
def explication3():
    """ Vizualisation of our work during MOPSI project """

    current_user.tuto3=True
    if current_user.tuto3:
        print("L'uilisateur a fait le tuto3")
    return render_template('explication3.html', title='Explanation')

@app.route('/explication/4', methods=['GET', 'POST'])
def explication4():
    """ Display the last step of the tutorial 
        The user can choose the atoms and plot the correcponding Rama plot """

    current_user.tuto4=True
    if current_user.tuto4:
        print("L'uilisateur a fait le tuto4")
    url=fetch_url("dialanine")

    if request.method == 'POST':
        try :
            phi_atom = [int(request.form['phi_atom1']), int(request.form['phi_atom2']), int(request.form['phi_atom3']), int(request.form['phi_atom4'])]
            psi_atom = [int(request.form['psi_atom1']), int(request.form['psi_atom2']), int(request.form['psi_atom3']), int(request.form['psi_atom4'])]
            
        except ValueError:
            return render_template('explication4.html', title='Explanation', url=url, error=True, completed_form=False)
        
        fig = rama.rama_plot(phi_atom, psi_atom)
        fig.write_html('static/templates/rama_user.html',full_html=False,include_plotlyjs='cdn')
        anim_fig = rama.rama_frame(phi_atom, psi_atom)
        anim_fig.write_html('static/templates/rama_frame.html', full_html=False,include_plotlyjs='cdn')
        return render_template('explication4.html', title='Explanation', url=url, error=False, completed_form=True)  

    return render_template('explication4.html', title='Explanation', url=url, error=False, completed_form=False)

@app.route('/explication/codeNN')
def codeAE():
    """ Show one of the notebooks we use in our MOPSI project
        This show thetraining for a 3D trajectory  """

    return render_template('TrainingAE.html')


if __name__ == "__main__":
    db.create_all()
    app.run(debug=True)