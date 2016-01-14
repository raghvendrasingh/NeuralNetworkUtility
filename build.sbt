name := "NeuralNetworkUtility"

version := "1.0"

scalaVersion := "2.11.7"

libraryDependencies ++= Seq(
  "org.scalanlp" % "breeze_2.11" % "0.11.2",
  "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly()
)

resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"