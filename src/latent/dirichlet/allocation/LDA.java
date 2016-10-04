/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package latent.dirichlet.allocation;

import com.hankcs.lda.Corpus;
import com.hankcs.lda.LdaGibbsSampler;
import com.hankcs.lda.LdaUtil;
import java.io.IOException;
import java.util.Map;

/**
 *
 * @author maryan
 */
public class LDA {
    
    public static void main(String args[]) throws IOException {
    // 1. Load corpus from disk
    Corpus corpus = Corpus.load("data/LDA_train");
    // 2. Create a LDA sampler
    LdaGibbsSampler ldaGibbsSampler = new LdaGibbsSampler(corpus.getDocument(), corpus.getVocabularySize());
    // 3. Train it
    
    ldaGibbsSampler.gibbs(5, 50/(double)4, 0.1);
    // 4. The phi matrix is a LDA model, you can use LdaUtil to explain it.
    double[][] phi = ldaGibbsSampler.getPhi();
    Map<String, Double>[] topicMap = LdaUtil.translate(phi, corpus.getVocabulary(), 20);
    LdaUtil.explain(topicMap);
    }
}
