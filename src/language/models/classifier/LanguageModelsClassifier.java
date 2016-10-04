/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package language.models.classifier;


import com.aliasi.classify.Classification;
import com.aliasi.classify.Classified;
//import com.aliasi.classify.ConfusionMatrix;
import com.aliasi.classify.DynamicLMClassifier;
import com.aliasi.classify.JointClassification;
import com.aliasi.classify.JointClassifier;
//import com.aliasi.classify.JointClassifierEvaluator;
//import com.aliasi.classify.LMClassifier;

import com.aliasi.lm.NGramProcessLM;

import com.aliasi.util.AbstractExternalizable;
import com.aliasi.util.Files;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 *
 * @author maryan
 */
public class LanguageModelsClassifier {

    /**
     * @param args the command line arguments
     */
    private static File TRAINING_DIR = new File("data/train");

//    private static File TESTING_DIR =  new File("data/test");
    
    private static String[] CATEGORIES = { 
        "harry.potter",
        "phone.firephone",
        "dark.tower.king",
        "fridge.freezer" 
    };

    private static int NGRAM_SIZE = 6;
    
    
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        
        DynamicLMClassifier<NGramProcessLM> classifier
            = DynamicLMClassifier.createNGramProcess(CATEGORIES, NGRAM_SIZE);

        for(int i=0; i<CATEGORIES.length; ++i) {
            File classDir = new File(TRAINING_DIR, CATEGORIES[i]);
            if (!classDir.isDirectory()) {
                String msg = "Could not find training directory="
                    + classDir;
                System.out.println(msg); // in case exception gets lost in shell
                throw new IllegalArgumentException(msg);
            }

            String[] trainingFiles = classDir.list();
            for (int j = 0; j < trainingFiles.length; ++j) {
                File file = new File(classDir,trainingFiles[j]);
                String text = Files.readFromFile(file,"ISO-8859-1");
                System.out.println("Training on " + CATEGORIES[i] + "/" + trainingFiles[j]);
                Classification classification = new Classification(CATEGORIES[i]);
                Classified<CharSequence> classified = new Classified<CharSequence>(
                        text, classification);
                classifier.handle(classified);
            }
        }
        
        
        //compiling
        System.out.println("Compiling");
        @SuppressWarnings("unchecked") // we created object so know it's safe
        JointClassifier<CharSequence> compiledClassifier
            = (JointClassifier<CharSequence>)
            AbstractExternalizable.compile(classifier);
 
        
       
        System.out.println("Write \'stop\' for termination ");
        String input = "";
        while(!input.equals("stop")) {

            InputStreamReader inputStreamReader = new InputStreamReader(System.in);
            BufferedReader reader = new BufferedReader(inputStreamReader);
            System.out.println("Type: ");
            input = reader.readLine();


            JointClassification jc =
                compiledClassifier.classify(input);
            String bestCategory = jc.bestCategory();
            System.out.println("It is : " + bestCategory);
//                System.out.println(jc.toString());
            System.out.println("---------------");

        }
        
    }
    
}
