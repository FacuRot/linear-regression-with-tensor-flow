var x_vals = [];
var y_vals = [];

let a, b, c;

const learningRate = 0.2;
const optimizer = tf.train.adam(learningRate);

function setup() {
  createCanvas(400, 400);

  a = tf.scalar(random(-1, 1)).variable();
  b = tf.scalar(random(-1, 1)).variable();
  c = tf.scalar(random(-1, 1)).variable();
}

function predict(equis) {
  const x_tensor = tf.tensor1d(equis);

  const y_tensor = a
    .mul(x_tensor.square())
    .add(b.mul(x_tensor))
    .add(c);
  return y_tensor;
}

function loss(pred, label) {
  return pred
    .sub(label)
    .square()
    .mean();
}

function mousePressed() {
  var x = map(mouseX, 0, width, -1, 1);
  var y = map(mouseY, 0, height, 1, -1);

  x_vals.push(x);
  y_vals.push(y);
}

function draw() {
  tf.tidy(() => {
    if (x_vals.length > 0) {
      const y_labels = tf.tensor1d(y_vals);
      optimizer.minimize(() => loss(predict(x_vals), y_labels));
    }
  });

  background(0);
  stroke(255);

  for (var i = 0; i < x_vals.length; i++) {
    const pointX = map(x_vals[i], -1, 1, 0, width);
    const pointY = map(y_vals[i], -1, 1, height, 0);
    strokeWeight(8);
    point(pointX, pointY);
  }

  var lineX = [];
  for (var i = -1; i < 1.01; i += 0.05) {
    lineX.push(i);
  }
  var y = tf.tidy(() => predict(lineX));
  var lineY = y.dataSync();
  y.dispose();

  beginShape();
  noFill();
  stroke(255);
  strokeWeight(2);
  for (var i = 0; i < lineX.length; i++) {
    var xpoint = map(lineX[i], -1, 1, 0, width);
    var ypoint = map(lineY[i], -1, 1, height, 0);
    vertex(xpoint, ypoint);
  }
  endShape();
}
