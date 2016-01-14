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

  def sigmoidVector(v: DenseVector[Double]): DenseVector[Double] = {
    val a = for (x <- v) yield (1/(1+math.exp(x)))
    a
  }

  def sigmoidMatrix(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    val a = for (x <- m) yield (1/(1+math.exp(x)))
    a
  }

  def meanSquaredErrorCost(numSamples: Int, modelOutput: DenseVector[Double], trainingLabels: DenseVector[Double],
                           weightDecayParamater: Double, layerWeights: Vector[DenseMatrix[Double]]): Double = {
    var sum2 = 0.0
    layerWeights foreach (x => {sum2 = sum2 + sum(x)})
    val cost = 1.toDouble/(2*numSamples) * sum((modelOutput - trainingLabels) map (x => x*x)) + (weightDecayParamater/2)*sum2
    cost
  }

  /* def softmax(m: DenseMatrix[Double]): DenseMatrix[Double] = {

   }*/

  def makeDropNodes(n: Int, v: Vector[Int], p: Double): List[DenseMatrix[Double]] = {
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
      dropNodeIndicesPerHiddenLayer = dropNodeIndicesPerHiddenLayer :+ dropNodeIndices
    }
    dropNodeIndicesPerHiddenLayer
  }

  def softmaxMatrix(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    val numer = m map (x => math.exp(x))
    for (i <- 0 to m.cols-1) {
      val maximum = max(m(::,i))
      var sum = 0.0
      m(::,i) foreach (x => {sum = sum + math.exp(x-maximum)})
      numer(::,i) :*= 1.toDouble/(sum*math.exp(maximum))
    }
    numer
  }

  def printList(lis: List[DenseMatrix[Double]]): Unit = {
    for ( i <- lis.indices) {
      println(lis(i))
      println()
    }
  }
}
