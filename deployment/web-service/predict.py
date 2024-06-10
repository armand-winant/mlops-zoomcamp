import pickle
from flask import Flask, request, jsonify

with open('lin_reg.bin', 'rb') as f:
	(dv, model) = pickle.load(f)

def prepare_features(ride):
	features = {
		'PU_DO': f"{ride['PULocationID']}_{ride['DOLocationID']}",
		'trip_distance': ride['trip_distance']
	}

	return features

def predict(features):
	X = dv.transform([features])
	preds = model.predict(X)[0]

	return preds

app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
	ride = request.get_json()

	features = prepare_features(ride)
	pred = predict(features)

	result = {
		'duration': pred
	}

	return jsonify(result)

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=9696)