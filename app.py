from src.Sales_Marketing.pipelines.prediction_pipeline import CustomData,PredictPipeline

from flask import Flask,request,render_template,jsonify


app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template("index.html")


@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    
    else:
        data=CustomData(
            
            Item_Weight=float(request.form.get('Item_Weight')),
            Item_Fat_Content = float(request.form.get('Item_Fat_Content')),
            Item_Visibility = float(request.form.get('Item_Visibility')),
            Item_Type= float(request.form.get('Item_Type')),
            Item_MRP = float(request.form.get('Item_MRP')),
            Outlet_Years = float(request.form.get('Outlet_Years')),
            Outlet_Identifier = float(request.form.get('Outlet_Identifier')),
            Outlet_Size = request.form.get('Outlet_Size'),
            Outlet_Location_Type= request.form.get('Outlet_Location_Type'),
            Outlet_Type = request.form.get('Outlet_Type')
        )
        # this is my final data
        final_data=data.get_data_as_dataframe()
        
        predict_pipeline=PredictPipeline()
        
        pred=predict_pipeline.predict(final_data)
        
        result=round(pred[0],2)
        
        return render_template("result.html",final_result=result)

#execution begin
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)