var x_vals = [];
var y_vals = [];

var m, b;

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

function setup() {
  createCanvas(400, 400);

  // Inicializo m y b con valores random entre 0 y 1
  // Estos son los valores que van a tener que ser optimizados
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)).variable());
}

function loss(pred, label) {
  // pred son las predicciones
  // label son las y de cada x

  // Mean Square Root
  return pred
    .sub(label)
    .square()
    .mean();
}

function predict(equis) {
  // Transformo las x a tensores
  const x_tensor = tf.tensor1d(equis);

  const y_tensor = x_tensor.mul(m).add(b);

  return y_tensor;
}

function mousePressed() {
  var x = map(mouseX, 0, width, 0, 1);
  var y = map(mouseY, 0, height, 1, 0);

  x_vals.push(x);
  y_vals.push(y);
}

function draw() {
  tf.tidy(() => {
    if (x_vals.length > 0) {
      const y_tensor = tf.tensor1d(y_vals);
      optimizer.minimize(() => loss(predict(x_vals), y_tensor));
    }
  });

  background(0);
  stroke(255);

  for (var i = 0; i < x_vals.length; i++) {
    const pointX = map(x_vals[i], 0, 1, 0, width);
    const pointY = map(y_vals[i], 0, 1, height, 0);
    strokeWeight(8);
    point(pointX, pointY);
  }

  const xs = [0, 1];

  const ys = tf.tidy(() => predict(xs));
  let lineY = ys.dataSync();
  ys.dispose();

  const x1 = map(xs[0], 0, 1, 0, width);
  const x2 = map(xs[1], 0, 1, 0, width);

  const y1 = map(lineY[0], 0, 1, height, 0);
  const y2 = map(lineY[1], 0, 1, height, 0);

  strokeWeight(2);
  line(x1, y1, x2, y2);
}

