from flask import Flask, render_template
from violence_detection import violence_bp
from crime_prediction import crime_bp

app = Flask(__name__)

# Register blueprints
app.register_blueprint(violence_bp, url_prefix='/violence')
app.register_blueprint(crime_bp, url_prefix='/crime')

@app.route('/')
def home():
    return render_template('home.html')  

if __name__ == '__main__':
    app.run(debug=True)