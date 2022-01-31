from flask import Flask, url_for, redirect, render_template, request
import numpy as np
import potentials as pt
from datetime import datetime
from flask import Flask, url_for, redirect, render_template, request,  flash
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_login import UserMixin, login_user, LoginManager, logout_user, current_user
from flask_bcrypt import Bcrypt
import os
import requests

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
        errors = False
        molecule = request.form['molecule']
        try:
            beta = int(request.form['beta'])
        except ValueError:
            flash("Please insert a new beta, it must be a float (floats should be with a point (exp : 0.5 not 0,5)")
            errors = True
        num = int(request.form['number'])
        bowls=[]
        
        for i in range(1,num+1):
            coord="bowl"+str(i)+"xy"
            xy=request.form[coord]
            try:
                x0,y0=xy.split(";")
                try:
                    x0=float(x0)
                except ValueError:
                    flash(f"Please insert a new x coordinate in the high intensity area number {i}, it must be a float (floats should be with a point (exp : 0.5 not 0,5)")
                    errors = True            
                try:
                    y0=float(y0)
                except ValueError:
                    flash(f"Please insert a new y coordinate in the high intensity area number {i} , it must be a float (floats should be with a point (exp : 0.5 not 0,5)")
                    errors = True
            except ValueError:
                flash(f"Please respect the coordinates format in the high intensity area number {i}, it must be at the form x;y ")
                errors = True            
            
            r_id="bowl"+str(i)+"r"
            try:
                r=float(request.form[r_id])
            except ValueError:
                flash(f"Please insert a new radius in the high intensity area number {i}, it must be a float (floats should be with a point (exp : 0.5 not 0,5)")
                errors = True

            a_id="bowl"+str(i)+"a"
            try:
                a=float(request.form[a_id])
            except ValueError:
                flash(f"Please insert a new amplitude in the high intensity area number {i}, it must be a float (floats should be with a point (exp : 0.5 not 0,5)")
                errors = True
            
            if not errors :
                if abs(x0) > 2:
                    flash(f"Please choose a value for x in the interval [-2,2] in the high intensity area number {i}")
                    errors = True
                if abs(y0) > 2:
                    flash(f"Please choose a value for x in the interval [-2,2] in the high intensity area number {i}")
                    errors = True
                bowls.append([x0,y0,r,a])

        url = fetch_url(molecule)
        response = requests.get(url)
        if response.status_code != 200:
            flash(f"The molecule {molecule} you looked for seems as if it doesn't exist in the database. Make sure it is spelled correctly !")
            errors = True

        if errors : 
            return render_template('visualisation.html', title='Visualisation', formulaire_rempli=False)
        else:
            bowl=np.array(bowls)
            potential=pt.MultimodalPotential(bowl, beta)
            fig_pot=pt.create_plots(potential)
            now = datetime.now() # current date and time to identify plots
            date_time = now.strftime("%m%d%Y%H%M%S%f")
            path_plot_pot ='static\\plot\\plot3D'+date_time+'.html' #path to 3D potential plot
            path_plot_pot= os.path.join(basedir,path_plot_pot)
            path_plot_trajectory = 'static\\img\\traj'+date_time+'.png' # path to trajectory plot
            path_plot_trajectory = os.path.join(basedir, path_plot_trajectory )
            file1 = open(path_plot_pot, 'w') #creating the files where we stores the plots
            file1.close()
            file2 = open(path_plot_trajectory, 'w')
            file2.close()
            fig_pot.write_html(path_plot_pot)
            fig_traj=pt.plot_trajectory(potential) 
            fig_traj.savefig(path_plot_trajectory)
            return render_template('visualisation.html', title='Visualisation', molecule=molecule , url=url, pot_path='/static/plot/plot3D'+date_time+'.html', traj_path='/static/img/traj'+date_time+'.png', formulaire_rempli = True)
 
    return render_template('visualisation.html', title='Visualisation', formulaire_rempli=False)

@app.route('/explication', methods=['GET','POST'])
def explication():
    current_user.change_tuto(0)
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
    current_user.tuto1=True
    if current_user.tuto1:
        print("L'uilisateur a fait le tuto1")
    return render_template('explication1.html', title='Explanation')

@app.route('/explication/2')
def explication2():
    current_user.tuto2=True
    if current_user.tuto2:
        print("L'uilisateur a fait le tuto2")
    return render_template('explication2.html', title='Explanation')

@app.route('/explication/3')
def explication3():
    current_user.tuto3=True
    if current_user.tuto3:
        print("L'uilisateur a fait le tuto3")
    return render_template('explication3.html', title='Explanation')

@app.route('/explication/4', methods=['GET', 'POST'])
def explication4():
    current_user.tuto4=True
    if current_user.tuto4:
        print("L'uilisateur a fait le tuto4")
    if request.method == 'POST':
        phi_atom1 = request.form['phi_atom1']
        phi_atom2 = request.form['phi_atom2']
        phi_atom3 = request.form['phi_atom3']
        phi_atom4 = request.form['phi_atom4']
       # phi_angle = np.array([int(phi_atom1), int(phi_atom2), int(phi_atom3), int(phi_atom4)])
    return render_template('explication4.html', title='Explanation')

@app.route('/explication/codeNN')
def codeAE():
    return render_template('TrainingAE.html')


if __name__ == "__main__":
    db.create_all()
    app.run(debug=True)