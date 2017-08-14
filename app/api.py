import logging

from flask import Flask, request, jsonify

import humans

app = Flask(__name__)


@app.route("/", methods=["POST"])
def recieve():
    fn = "/tmp/humanspic.png"
    with open(fn, "wb") as f:
        f.write(request.stream.read())
    matches = humans.get_people(open(fn), "rb", session)
    return jsonify(matches)


if __name__ == "__main__":
    logging.debug("Starting App")
    with humans.detection_graph.as_default():
        with humans.tf.Session(graph=humans.detection_graph) as session:
            app.run(debug=True)
