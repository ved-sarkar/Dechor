from shiny import App, render, ui
from shiny import App, render, ui
from shiny import *
from shiny.types import ImgData

import pickle
import numpy as np
import pandas as pd

app_ui = ui.page_fluid(

    ui.h2("DECHOR: a Decision trEe for Chronic subdural HematOma Referral outcome prediction"),
    ui.h6("Please fill in the following form"),
    ui.layout_sidebar(
        ui.panel_sidebar(
                         ui.input_slider("age", "Age", 0, 100, 40),
                         ui.input_select("headache", label = "Headache", choices = {"0": "No", "1" : "Yes"}, selected = "No"),
                         ui.input_select("dementia", label = "Dementia", choices = {"0": "No", "1": "Yes"}, selected = "No"),
                         ui.input_select("motor_weakness", label = "Motor Weakness", choices = {"0": "No", "1": "Yes"}, selected = "No"),
                         ui.input_select("midline_shift", label = "Midline Shift", choices = {"0": "No", "1": "Yes"}, selected = "No"),
                         ui.input_select("CSDHsize", label = "Size of CSDH", choices = {"1": "Small", "2": "Medium", "3": "Large"}, selected = "Small"),
                         ui.input_select("QoL", label = "Pre-morbid QoL", choices = {"0": "Reasonable", "1": "Poor"}, selected = "Reasonable")
                         ),

        ui.panel_main(
            ui.output_text_verbatim("score"),
            ui.output_image("image1")
           
            ),
    ),
)



dt = 'dt.pkl'
model = pickle.load(open(dt, 'rb'))

scalerfile = 'scaler.sav'
scaler = pickle.load(open(scalerfile, 'rb'))




def server(input, output, session):
    @output
    @render.text
    def score():
        #Store the input parameters
        userinput_age = int(input.age())
        userinput_headache = int(input.headache())
        userinput_dementia = int(input.dementia())
        userinput_motor = int(input.motor_weakness()) 
        userinput_midline = int(input.midline_shift()) 
        userinput_size = int(input.CSDHsize()) 
        userinput_qol = int(input.QoL())
        #Normalize the "Age" input parameter
        userinput_age = np.reshape(userinput_age, (1,-1))
        test_scaled_set = scaler.transform(userinput_age)
        #Creat the final input array
        test = [test_scaled_set, userinput_headache, userinput_dementia, userinput_motor, userinput_midline, userinput_size, userinput_qol]
        finalArray = np.asarray(test, dtype = np.float64, 
                            order ='C')
        henry = np.transpose(finalArray)
        test= pd.DataFrame(data = henry, index=['age','headache' ,'dementia','Motor weakness', 'midline shift', 'CSDH size','Pre-morbid QoL']) 
        test = test.T
        #Predict the output
        acceptance = model.predict(test)
        prediction = model.predict_proba(test)[:, 1]
        
        return "Acceptance status: " + f"{acceptance}" +"\nPrediction: " + f"{prediction}"
    
    @output
    @render.image 
    def image1():
        from pathlib import Path

        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": str(dir / "decision_tree1.png"), "width": "100%"}
        return img 




    
app = App(app_ui, server)