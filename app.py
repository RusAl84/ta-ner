from flask import Flask, render_template, url_for, request, redirect
from flask_cors import CORS, cross_origin
import defrazmeka
import defner
from flask import send_from_directory



host = "http://127.0.0.1:5003"

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/static/<path:path>')
def send_report(path):
    return send_from_directory('static', path)

@app.route('/', methods =['POST', 'GET'])
@app.route('/analyse', methods =['POST', 'GET'])
def analyse():
    if request.method == "POST":
        fileAnaliz = request.form['dir']
        newFilename = request.form['wfile']
        crfmodel = request.form['crfmodel']
        hand_text=""
        hand_text = request.form['hand_text']

        if len(str(hand_text))>10:
            defrazmeka.razmetka_text(hand_text)
            defner.analiz("tmp.csv", newFilename, crfmodel)
        else:
            defner.analiz(fileAnaliz, newFilename, crfmodel)
        return redirect(host+"/static/"+newFilename+".csv", code=302) #render_template("analyse.html")
    else:
        return render_template("analyse.html")


@app.route('/textprep', methods =['POST', 'GET'])
def textprep():
    if request.method == "POST":
        filename = request.form['wfile']
        file = request.form['dir']

        defrazmeka.razmetka(file, filename)
        return redirect(host+"/static/"+filename+".csv", code=302) #render_template("textprep.html")
    else:
        return render_template("textprep.html")


@app.route('/learning', methods =['POST', 'GET'])
def learning():
    if request.method == "POST":
        file = request.form['dir']
        #filename = request.form['wfile']
        defner.learning(file)
        return redirect(host+"/static/crf.pickle", code=302) #render_template("learning.html")
    else:
        return render_template("learning.html")


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0", port="5003")

