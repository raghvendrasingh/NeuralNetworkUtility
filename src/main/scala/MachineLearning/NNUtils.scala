package MachineLearning

/**
  * Created by raghvendra.singh on 1/9/16.
  */

import _root_.breeze.linalg.DenseMatrix
import _root_.breeze.linalg.DenseVector
import _root_.breeze.linalg.Vector
import breeze.linalg._
import scala.util.Random


object NNUtils {

  /** Defining global variables */
  private var layerWeights = List[DenseMatrix[Double]]()
  private var layerBiases = List[DenseVector[Double]]()
  private var layerOutputs = List[DenseMatrix[Double]]()
  private var deltaLayerWeights = List[DenseMatrix[Double]]()
  private var deltaLayerBiases = List[DenseVector[Double]]()
  private var numNodesInLayers = Vector[Int]()
  private var dropoutProbability = 0.0

  /** This method initializes layer weights, layer biases and num nodes per layer of the neural net
    *
    * @param nodesPerLayer - This is a vector containing the number of nodes in each layer of the neural net.
    */
  def initializeNet(nodesPerLayer: Vector[Int]): Unit = {
    numNodesInLayers = nodesPerLayer
    if (numNodesInLayers.size < 2) {
      throw new Exception("Not enough layers in neural network. Please provide atleast three layers including input and output layer.")
    }
    for (i <- 0 to numNodesInLayers.size - 2) {
      val b = math.sqrt(6.toDouble / (numNodesInLayers(i) + numNodesInLayers(i + 1)))
      val vec = DenseVector.zeros[Double](numNodesInLayers(i) * numNodesInLayers(i + 1)) map (x => math.random * (b + b) - b)
      val weight = new DenseMatrix[Double](numNodesInLayers(i), numNodesInLayers(i + 1), vec.toArray)
      layerWeights = layerWeights :+ weight
      val bias: DenseVector[Double] = DenseVector.rand(numNodesInLayers(i + 1))
      layerBiases = layerBiases :+ bias
    }
    assert(layerWeights.size == numNodesInLayers.size - 1)
    assert(layerBiases.size == numNodesInLayers.size - 1)
  }

  /** This method prints layer weights and layer biases. */
  def printLayerWeightsAndLayerBiases(): Unit = {
    for (i <- layerWeights.indices) {
      println(s"Layer ${i} - ${i + 1} weights=")
      println(layerWeights(i))
      println(s"Layer ${i} - ${i + 1} biases=")
      println(layerBiases(i))
    }
  }

  /** This method calculates the mean squared error cost
    *
    * @param numSamples - Number of training samples.
    * @param trainingData - Training data matrix where each column represents one sample and no. of rows = no. of features per training sample.
    * @param trainingLabels - Training label matrix where each column represents one output sample and no. of rows = no. of output nodes in neural net.
    * @param weightDecayParameter - Regularization parameter to avoid overfitting.
    * @param params - Serialized weights and biases.
    * @return - Returns the mean squared error cost.
    */
  private def meanSquaredErrorCost(numSamples: Int, trainingData: DenseMatrix[Double], trainingLabels: DenseMatrix[Double],
                           weightDecayParameter: Double, params: DenseVector[Double]): Double = {
    val thetaTup = NNUtils.deserializeParams(numNodesInLayers, params)
    val weightList = thetaTup._1
    val biasList = thetaTup._2
    var out = trainingData
    for (i <- 0 to numNodesInLayers.size - 2) {
      var tempOut: DenseMatrix[Double] = weightList(i).t * out
      tempOut = tempOut(::, *) + biasList(i)
      out = NNUtils.sigmoidMatrix(tempOut)
    }
    /** Calculate output for output layer using softmax function */
    /*var softmaxInput: DenseMatrix[Double] = weightList.last.t * out
    softmaxInput = softmaxInput(::, *) + biasList.last
    val output = NNUtils.softmaxMatrix(softmaxInput)*/

    var sum2 = 0.0
    for (i <- weightList.indices) {
      val m = weightList(i) map (x => x*x)
      sum2 = sum2 + sum(m)
    }
    val cost = ((1.toDouble/(2*numSamples)) * sum((out - trainingLabels) map (x => x*x))) + (weightDecayParameter/2)*sum2
    cost
  }

  /** This method computes the numerical gradient using the mean squared error cost
    *
    * @param numSamples - No. of training samples.
    * @param trainingData - Training data matrix where each column represents one sample and no. of rows = no. of features per training sample.
    * @param trainingLabels - Training label matrix where each column represents one output sample and no. of rows = no. of output nodes in neural net.
    * @param weightDecayParameter - Regularization parameter to avoid overfitting.
    * @param params - Serialized weights and biases.
    * @return - Returns the numerical gradient calculated at theta = params.
    */
  private def computeNumericalGradient(numSamples: Int, trainingData: DenseMatrix[Double], trainingLabels: DenseMatrix[Double],
                               weightDecayParameter: Double, params: DenseVector[Double]): DenseVector[Double] = {
    val m = params.size
    val numGrad = DenseVector.zeros[Double](m)
    val eps = 0.0001
    for (i <- 0 to m-1) {
      val q = DenseVector.zeros[Double](m)
      q(i) = q(i) + eps
      numGrad(i) = ( meanSquaredErrorCost(numSamples, trainingData, trainingLabels, weightDecayParameter, params+q) - meanSquaredErrorCost(numSamples, trainingData, trainingLabels, weightDecayParameter, params-q) )/(2*eps)
    }
    numGrad
  }

  /** This method compares the gradient calculated using back propagation with the numerical gradient.
    *
    * @param grad1 - gradient calculated using back propagation.
    * @param grad2 - numerical gradient.
    */
  private def checkNumericalGradient(grad1: DenseVector[Double], grad2: DenseVector[Double]): Unit = {
    assert(grad1.size == grad2.size)
    for (i <- 0 to grad1.size-1) println(grad1(i) + "  " + grad2(i) )

    val res = norm(grad1-grad2)/norm(grad1+grad2)
    println("The norm of difference between numerical and analytical gradient is = "+ res)
    if (res > 1e-8) println("Gradient calculated through backpropagation is not correct.")
  }

  /** This method prints deltaLayerWeights and deltaLayerBiases. */
  def printDeltaLayerWeightsAndDeltaLayerBiases(): Unit = {
    for (i <- deltaLayerWeights.indices) {
      println(s"delta layer ${i} - ${i + 1} weights=")
      println(deltaLayerWeights(i))
      println(s"delta layer ${i} - ${i + 1} biases=")
      println(deltaLayerBiases(i))
    }
  }

  /** This method performs the forward pass in the neural network populating layerOutputs. */
  private def forward(input: DenseMatrix[Double], dropNodeList: List[DenseMatrix[Double]]): Unit = {
    layerOutputs = layerOutputs.drop(layerOutputs.size)
    layerOutputs = layerOutputs :+ input
    for (i <- 0 to numNodesInLayers.size - 3) {
      assert(layerWeights(i).t.cols == layerOutputs(i).rows)
      var tempOut: DenseMatrix[Double] = layerWeights(i).t * layerOutputs(i)
      tempOut = tempOut(::, *) + layerBiases(i)
      layerOutputs = layerOutputs :+ (NNUtils.sigmoidMatrix(tempOut) :* dropNodeList(i))
    }
    val temp: DenseMatrix[Double] = layerWeights.last.t * layerOutputs.last
    layerOutputs = layerOutputs :+ NNUtils.sigmoidMatrix( temp(::,*) + layerBiases.last)
    /** Calculate output for output layer using softmax function */
    /*var softmaxInput: DenseMatrix[Double] = layerWeights.last.t * layerOutputs.last
    softmaxInput = softmaxInput(::, *) + layerBiases.last
    val output = NNUtils.softmaxMatrix(softmaxInput)
    layerOutputs = layerOutputs :+ output*/
  }

  /** This method performs the backward pass in the neural network populating deltaLayerWeights and deltaLayerBaises. */
  private def backward(numSamples: Int, targetLabels: DenseMatrix[Double], dropNodeList: List[DenseMatrix[Double]], weightDecay: Double): Unit = {
    val outputDelta = -(targetLabels - layerOutputs.last) :* (layerOutputs.last :* (DenseMatrix.ones[Double](layerOutputs.last.rows, layerOutputs.last.cols) - layerOutputs.last))
    var delta = outputDelta
    val size = numNodesInLayers.size - 2
    val tempDeltaWeight: DenseMatrix[Double] = layerOutputs(size) * delta.t
    val deltaWeight: DenseMatrix[Double] = ((1.toDouble/numSamples) * tempDeltaWeight)  + (layerWeights.last * weightDecay)
    deltaLayerWeights = deltaWeight +: deltaLayerWeights
    deltaLayerBiases = ((1.toDouble/numSamples) * sum(delta(*, ::))) +: deltaLayerBiases
    for (i <- numNodesInLayers.size - 2 to 1 by -1) {
      val temp1: DenseMatrix[Double] = layerWeights(i) * delta
      val hiddenDelta = temp1 :* (layerOutputs(i) :* (DenseMatrix.ones[Double](layerOutputs(i).rows, layerOutputs(i).cols) - layerOutputs(i)))
      delta = hiddenDelta
      val temp2: DenseMatrix[Double] = layerOutputs(i - 1) * delta.t
      val temp3: DenseMatrix[Double] = ((1.toDouble/numSamples) * temp2) + (layerWeights(i-1) * weightDecay)
      deltaLayerWeights = temp3 +: deltaLayerWeights
      val temp4: DenseMatrix[Double] = (delta :* dropNodeList(i - 1)) * (1.toDouble/numSamples)
      deltaLayerBiases = sum(temp4(*, ::)) +: deltaLayerBiases
    }
  }

  /** This method prints layerOutputs. */
  def printLayerOutputs(): Unit = {
    println("Layer outputs:")
    println()
    for (i <- layerOutputs.indices) {
      println(layerOutputs(i))
      println()
    }
  }

  /** This method calculates the sigmoid of each value in the passed DenseVector.
    *
    * @param v - This is a passed DenseVector.
    * @return - Returns the sigmoid DenseVector of passed DenseVector.
    */
  private def sigmoidVector(v: DenseVector[Double]): DenseVector[Double] = {
    val a = for (x <- v) yield 1/(1+math.exp(-x))
    a
  }

  /** This method calculates the sigmoid of each value in the passed DenseMatrix
    *
    * @param m - This is a passed DenseMatrix.
    * @return - Returns the sigmoid DenseMatrix of passed DenseMatrix.
    */
  private def sigmoidMatrix(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    val a = for (x <- m) yield 1/(1+math.exp(-x))
    a
  }



  /** This method returns a list of DenseMatrix where a column in a DenseMatrix represents drop nodes for a hidden layer.
    * Each column contains 0's and 1's where 0 means that the node in this hidden layer is dropped out and 1 means it is not.
    * For ex. If we have a List of DenseMatrix like [DenseMatrix[(1,0),(0,1),(1,1)],DenseMatrix[(1,1),(0,0)],DenseMatrix[(0,1),(1,0)]].
    * The first DenseMatrix represents the drop nodes for first hidden layer for 2 training samples. For the first training sample
    * and for the first hidden layer we have first node present, second node dropped out and third node present. For the second training
    * sample and for first hidden layer we have first node dropped out but both second and third node present.
    *
    * @param n - Number of training samples in the dataset.
    * @param v - This is a vector of number of nodes in each layer of the neural network.
    * @param p - It is the dropout probability.
    * @param isDropOut - This is a boolean flag to represent If we want to perform dropout or not.
    * @return - A list of DenseMatrix representing drop out nodes in each layer and for each training sample in data set.
    */
  private def makeDropNodes(n: Int, v: Vector[Int], p: Double, isDropOut: Boolean): List[DenseMatrix[Double]] = {
    var dropNodeIndicesPerHiddenLayer = List[DenseMatrix[Double]]()
    for (i <- 1 to v.size - 2) {
      val dropNodeIndices: DenseMatrix[Double] = DenseMatrix.ones(v(i), n)
      val rows = dropNodeIndices.rows
      val cols = dropNodeIndices.cols
      for (j <- 0 to cols-1) {
        for (k <- 0 to rows - 1) {
          val rnd = Random.nextDouble()
          if (rnd < p) dropNodeIndices(k,j) = 0.0
        }
        if (sum(dropNodeIndices(::,j)) == 0.0) dropNodeIndices(Random.nextInt(v(i)),j) = 1.0
        else if (sum(dropNodeIndices(::,j)) == v(i)) dropNodeIndices(Random.nextInt(v(i)),j) = 0.0
      }
      if (isDropOut) dropNodeIndicesPerHiddenLayer = dropNodeIndicesPerHiddenLayer :+ dropNodeIndices
      else dropNodeIndicesPerHiddenLayer = dropNodeIndicesPerHiddenLayer :+ DenseMatrix.ones[Double](v(i), n)
    }
    dropNodeIndicesPerHiddenLayer
  }

  /** This method calculates the softmax of each value in the passed DenseMatrix
    *
    * @param m - This is a passed DenseMatrix.
    * @return - Returns the softmax DenseMatrix of passed DenseMatrix.
    */
  private def softmaxMatrix(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    val numerator = m map (x => math.exp(x))
    for (i <- 0 to m.cols-1) {
      val maximum = max(m(::,i))
      var sum = 0.0
      m(::,i) foreach (x => {sum = sum + math.exp(x-maximum)})
      numerator(::,i) :*= 1.toDouble/(sum*math.exp(maximum))
    }
    numerator
  }

  /** This method serialzes layer weights and layer biases in a DenseVector in column major order.
    *
    * @param params1 - layer weights.
    * @param params2 - layer biases.
    * @return - Returns a DenseVector of all the serialized parameters.
    */
  private def serializeParams(params1: List[DenseMatrix[Double]], params2: List[DenseVector[Double]]): DenseVector[Double] = {
    /** Always serialize like col1,col2,col3,....,coln */
    var result = DenseVector[Double]()
    for (i <- params1.indices) result = DenseVector.vertcat(result, params1(i).toDenseVector)
    for (i <- params2.indices) result = DenseVector.vertcat(result, params2(i))
    result
  }

  /** This method deserializes a parametr vector into layer weights and layer baises
    *
    * @param numNodesInLayers - This is a vector representing no. of nodes in each layer of the neural net.
    * @param params - This is a parameter vector containing the layer weights and layer bias values in column major order.
    * @return - Returns a tuple of layer weights and layer biases.
    */
  private def deserializeParams(numNodesInLayers: Vector[Int], params: DenseVector[Double]): (List[DenseMatrix[Double]], List[DenseVector[Double]]) = {
    var tempWeightsList = List[DenseMatrix[Double]]()
    var tempBiasList = List[DenseVector[Double]]()
    var start = 0
    var end = 0
    for (i <- 0 to numNodesInLayers.size-2) {
      end = start + numNodesInLayers(i)*numNodesInLayers(i+1)
      tempWeightsList = tempWeightsList :+ reshape(params.slice(start,end), numNodesInLayers(i), numNodesInLayers(i+1))
      start = end
    }
    for (i <- 0 to numNodesInLayers.size-2) {
      end = start + numNodesInLayers(i+1)
      tempBiasList = tempBiasList :+ params.slice(start,end)
      start = end
    }
    (tempWeightsList, tempBiasList)
  }

  /** This method updates the layer weights and layer biases of the neural net according to some optimization algorithm.
    *
    * @param numSamples - Number of training samples.
    * @param learningRate - Learning rate used in optimization algorithm.
    * @param weightDecay - Regularization parameter to avoid overfitting.
    */
  private def updateWeights(numSamples: Int, learningRate: Double, weightDecay: Double): Unit = {
    var newLayerWeights = List[DenseMatrix[Double]]()
    var newLayerBiases = List[DenseVector[Double]]()
    try {
      for (i <- layerWeights.indices) {
        deltaLayerWeights(i) :*= (1.toDouble / numSamples)
        val temp3: DenseMatrix[Double] = deltaLayerWeights(i) + (layerWeights(i) * weightDecay)
        val temp4 = layerWeights(i) - (temp3 * learningRate)
        newLayerWeights = newLayerWeights :+ temp4
        newLayerBiases = newLayerBiases :+ (layerBiases(i) - (deltaLayerBiases(i) :*= (learningRate / numSamples)))
      }
      layerWeights = newLayerWeights
      layerBiases = newLayerBiases
    } catch {
      case ex: Exception => {
        println("exception is:", ex)
        throw new Exception("Unexpected execution error while executing method updateWeights()", ex)
      }
    }
  }

  /** This method trains the neural net
    *
    * @param trainingSamples - Training data with one sample per column. No. of rows = No. of features per sample.
    * @param trainingLabels - Training output with one output sample per column. No. of rows = No. of output units in neural net.
    * @param maxEpochs - Maximum no. of iterations.
    * @param learningRate - Learning rate used in the optimization algorithm.
    * @param weightDecay - It is a regularization parameter to avoid overfitting in neural net.
    * @param checkNumericalGrad - This a boolean flag to indicate numerical gradient check.
    * @param isDropOut - This is a boolean flag to indicate use of dropout neural net.
    * @param dropoutProb - This is a dropout probability. Each node in hidden layer can be chucked out with this probability.
    */

  def train(trainingSamples: DenseMatrix[Double], trainingLabels: DenseMatrix[Double], maxEpochs: Int,
            learningRate: Double, weightDecay: Double = 0.0, checkNumericalGrad: Boolean = false, isDropOut: Boolean = false, dropoutProb: Double = 0.0): Unit = {
    dropoutProbability = dropoutProb
    var epoch = 0
    val numSamples = trainingSamples.cols
    var k = 1
    while (epoch < maxEpochs) {
      if (epoch == k * 100) {
        println(s"${k * 100} epochs completed.")
        k = k + 1
      }
      deltaLayerWeights = List[DenseMatrix[Double]]()
      deltaLayerBiases = List[DenseVector[Double]]()
      val dropNodeList = makeDropNodes(numSamples, numNodesInLayers, dropoutProbability, isDropOut)
      val shuffledTuple = shuffleMatrix(trainingSamples, trainingLabels)
      forward(shuffledTuple._1, dropNodeList)
      backward(numSamples, shuffledTuple._2, dropNodeList, weightDecay)
      if (checkNumericalGrad) {
        val numGrad = computeNumericalGradient(numSamples, shuffledTuple._1, shuffledTuple._2, weightDecay, NNUtils.serializeParams(layerWeights, layerBiases))
        checkNumericalGradient(serializeParams(deltaLayerWeights, deltaLayerBiases), numGrad)
        epoch = 1
      }
      updateWeights(numSamples, learningRate, weightDecay)
      epoch = epoch + 1
    }
  }

  /** This method shuffles the column of the passed training sample matrix and training labels matrix.
    *
    * @param m - This is a training sample matrix.
    * @param v - This is a training label matrix.
    * @return - Returns a tuple of shuffled training sample matrix and training label matrix.
    */
  private def shuffleMatrix(m: DenseMatrix[Double], v: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    var sequence = List[Int]()
    for (i <- 0 to m.cols - 1)
      sequence = sequence :+ i
    sequence = Random.shuffle(sequence)
    val res1 = DenseMatrix.zeros[Double](m.rows,m.cols)
    val res2 = DenseMatrix.zeros[Double](v.rows,v.cols)
    var k = 0
    for (i <- sequence) {
      res1(::,k) := m(::,i)
      res2(::,k) := v(::,i)
      k = k + 1
    }
    (res1, res2)
  }
}
