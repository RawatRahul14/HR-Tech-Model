from flask import Flask , render_template , request
from flask_pymongo import PyMongo
from werkzeug.utils import secure_filename
import os 

app = Flask(__name__)
# app.config["MONGO_URI"] = "mongodb://localhost:27017/myDatabase"
# mongo = PyMongo(app)

@app.route('/')  # route to display the home page
def homePage():
    return render_template("index.html")

app.run(debug=True , port=5001)


@app.route('/upload',methods=['PUT','GET'])
def index():
    if request.method == 'PUT':
        try:
            #  reading the inputs given by the user
            name =request.form['name']
            contact_number =request.form['contact_number']
            email_id =request.form['email_id']
            def upload_file():
                 if 'pdfFile' not in request.files:
                      return 'No file part'
                 file = request.files['pdfFile']
                 if file.filename == '':
                      return 'No selected file'
                 if file:
                      filename = secure_filename(file.filename)
                      file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))# Save the filename to the database
                      return 'File uploaded successfully'
                 if __name__ == '__main__':
                     app.run() 

        except Exception as e:
            print('The Exception message is: ',e)
            
            return 'something is wrong'
        else:
            return render_template('index.html')
        if __name__ == "__main__":
            app.run(debug=True , host="0.0.0.0", port = 8080)
