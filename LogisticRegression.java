import java.io.*;
import java.sql.Array;
import java.util.*;
import java.util.Collections;

/**
 * Aditi Jorapur
 * aditi.jorapur@sjsu.edu
 * 015617225
 * Class to implement Logistic Regression (without bias) for a Spam Detection System.
 * In order to run the code replace the paths in the main method with the local paths for the
 * train and test data set files
 */

public class LogisticRegression {
    /** the learning rate */
    private double rate= 0.01;
    /** the weights to learn */
    private double[] weights;
    /** the number of iterations */
    private int ITERATIONS = 200;

    /**
     * Constructor initializes the weight vector and sets the weights array to vector 0
     * @param data the dataset, we will take the length of the dataset to find the number of features
     */
    public LogisticRegression(double[][] data){
        //make the weights array the length of the data set
        this.weights = new double[data[0].length - 1];
        //fill in the weights array with 0.0
        Arrays.fill(weights, 0.0);
    }

    /** Implement the sigmoid function **/
    public double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * Helper function for prediction.
     * Takes a test instance as input and outputs the probability of the label being 1.
     * This function calls sigmoid()
     * @param feature The features of the instance
     * @return The predicted probability of the label being 1.
     */
    public double predictionHelper(double[] feature) {
        double x = 0.0;
        for (int i = 0; i < feature.length; i++) {
            x += feature[i] * weights[i];
        }
        return sigmoid(x);
    }

    /**
     * The prediction function.
     * Takes a test instance as input and outputs the predicted label.
     * This function calls the predictionHelper method.
     * @param feature The features of the instance
     * @return The predicted label which is 0 or 1
     */
    public int prediction(double[] feature) {
        // check if the value is above or below 0.5 and then return 1 or 0
        return (predictionHelper(feature) >= 0.5) ? 1 : 0;
    }

    /**
     * This function takes a test set as input, calls the predict function to predict a label for it,
     * and prints the accuracy, P, R, and F1 score of the positive class and negative class and the confusion matrix.
     * @param data the data set
     */
    public void printAccuracy(double[][] data){
        double truePositive = 0;
        double trueNegative = 0;
        double falsePositive = 0;
        double falseNegative = 0;

        //gets an array of the labels
        double[] labels = new double[data.length];
        double avg = 0.0;
        for (int i = 0; i < data.length; i++) {
            labels[i] = data[i][data[i].length - 1];
        }

        //calculate the truePositive, falsePositive, falseNegative, trueNegative
        for (int i = 0; i < data.length; i++) {
            double predict = prediction(Arrays.copyOf(data[i], data[i].length - 1));
            if (predict == 1 && labels[i] == 1) {
                truePositive++;
            } else if (predict == 1 && labels[i] == 0) {
                falsePositive++;
            } else if (predict == 0 && labels[i] == 1) {
                falseNegative++;
            } else {
                trueNegative++;
            }
        }

        //calculate the accuracy
        double accuracy = (truePositive + trueNegative) / data.length;
        System.out.println("Accuracy: " + accuracy);
        System.out.println(" ");

        //calculate the precision
        double posPre = truePositive / (truePositive + falsePositive);
        double negPre = trueNegative / (trueNegative + falseNegative);

        //calculate the recall
        double posRecall = truePositive / (truePositive + falseNegative);
        double negRecall = trueNegative / (trueNegative + falsePositive);

        //calculate the F1 score
        double posF1 = 2 * (posPre * posRecall) / (posPre + posRecall);
        double negF1 = 2 * (negPre * negRecall) / (negPre + negRecall);

        //print out the positive class
        System.out.println("~~~ Positive Class ~~~");
        System.out.println("Positive Class Precision - Spam: " + posPre);
        System.out.println("Positive Class Recall - Spam: " + posRecall);
        System.out.println("Positive Class F1 Score - Spam: " + posF1);
        System.out.println(" ");

        //print out the negative class
        System.out.println("~~~ Negative Class ~~~");
        System.out.println("Negative Class Precision - Ham: " + negPre);
        System.out.println("Negative Class Recall - Ham: " + negRecall);
        System.out.println("Negative Class F1 Score - Ham: " + negF1);
        System.out.println(" ");

        //print out the confusion matrix
        System.out.println("~~~ Confusion Matrix ~~~");
        System.out.println("\t\t    Predicted");
        System.out.println("\t\t        Spam  \tHam");
        System.out.println("Actual\tSpam \t" + truePositive + "\t" + falseNegative);
        System.out.println("\t     Ham   \t" + falsePositive + "\t" + trueNegative);



    }

    /**
     * Train the Logistic Regression using Stochastic Gradient Descent and compute the log loss.
     * @param data The training data set
     */
    public void training(double[][] data) {
        // value for total cost of long loss
        double totalCost = 0.0;
        //loop through the data and make an array of the labels
        double[] labels = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            labels[i] = data[i][data[i].length - 1];
        }

        // Go through the set number of iterations
        for (int i = 0; i < ITERATIONS; i++) {
            //initialize loss
            double loss = 0.0;
            //update the weights
            for (int j = 0; j < data.length; j++) {
                //predict with the current val
                double predict = predictionHelper(Arrays.copyOf(data[j], data[j].length - 1));
                //calculate the error
                double error = labels[j] - predict;
                //go through the weights and update the weights with the error
                for (int k = 0; k < weights.length; k++) {
                    weights[k] += rate * error * data[j][k];
                }
            }
            //calculate the loss
            for (int k = 0; k < data.length; k++) {
                //use predict
                double predict = predictionHelper(Arrays.copyOf(data[k], data[k].length - 1));
                //calculate the loss and add to it
                loss += -labels[k] * Math.log(predict) - (1 - labels[k]) * Math.log(1 - predict);
            }

            // Normalize the log loss
            loss /= data.length;
            //add the loss to the totalCost
            totalCost += loss;
            // Print log loss for every iteration
            System.out.println("Iteration: " + (1 + i) + " Log Loss: " + loss);
            //System.out.println(loss);
        }

        // Print a blank line for viewing purposes
        System.out.println();

        //print the accuracy
        System.out.println("~~~ trainData Accuracy ~~~");
        printAccuracy(data);
        //divide the total cost by 200 (the number of iterations)
        totalCost /= 200;
        //print out the total cost
        System.out.println("Total Cost: " + totalCost);
    }


    /**
     * Function to read the input dataset.
     * @param filename name of the file we are reading in
     * @return the data in a 2D array
     * @throws FileNotFoundException if file is not found
     */
    public static double[][] readDataset(String filename) throws FileNotFoundException {
        //create a new 2D array
        List<double[]> temp = new ArrayList<>();
        //actual 2D array that will hold the values
        double[][] data = null;
        //set buffered reader to null
        BufferedReader br = null;

        try {
            br = new BufferedReader(new FileReader(filename));
            //skip the first line because it contains the labels (feature names)
            br.readLine();
            //make a new string for the line
            String line;

            //go through the row we are at
            while ((line = br.readLine()) != null) {
                //make an array for the values
                String[] values = line.trim().split(",");
                //create a new array
                double[] row = new double[values.length];
                //add the values to the array
                for (int i = 0; i < values.length; i++) {
                    row[i] = Double.parseDouble(values[i]);
                }

                // Add the row to the entire 2D array
                temp.add(row);
            }
            //populate the 2D array that will actually hold the values
            data = temp.toArray(new double[0][]);

        } catch (IOException e) {
            // throw exception
            throw new RuntimeException("Error reading file: " + e.getMessage());
        } finally {
            //close the buffered reader
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        return data;
    }



    /** main Function
     * readDataset uses local file path to test and training data
     * **/
    public static void main(String[] args) {
        try {
            //training data file -> put it in an array
            double[][] trainingData = readDataset("/Users/aditijorapur/Downloads/LogisticRegression/src/train-1.csv");

            //create a new instance with the features from the data
            LogisticRegression logisticRegression = new LogisticRegression(trainingData);

            // test data file -> put it in an array
            double[][] testData = readDataset("/Users/aditijorapur/Downloads/LogisticRegression/src/test-1.csv");

            //train the model with the training data
            logisticRegression.training(trainingData);

            //test data accuracy
            System.out.println(" ");
            System.out.println("~~~ testData Accuracy ~~~");
            logisticRegression.printAccuracy(testData);

        } catch (FileNotFoundException e) {
            System.err.println("Error reading dataset: " + e.getMessage());
        }
    }





}
