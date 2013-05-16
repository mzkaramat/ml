/**
 * Copyright (c) 2013, Cloudera, Inc. All Rights Reserved.
 *
 * Cloudera, Inc. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"). You may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * This software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for
 * the specific language governing permissions and limitations under the
 * License.
 */

package com.cloudera.science.ml.classifiers;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.cloudera.science.ml.core.vectors.LabeledVector;

public class ClassifierTrainer {
  private Random rand;
  // TODO: report progress

  public ClassifierTrainer(long seed) {
    rand = new Random(seed);
  }
  
  // The MIN_SCALING_FACTOR is used to protect against combinations of
  // lambda * eta > 1.0, which will cause numerical problems for regularization
  // and PEGASOS projection.
  private static final double MIN_SCALING_FACTOR = 0.0000001;

  public void fitClassifier(List<LabeledVector> trainingSet, ClassifierParams params) {
    MutableVector w = new MutableVector(params.getNumFeatures());
    fitClassifier(trainingSet, params.getLoopType(), params.getLearnerType(),
        params.getEtaType(), params.getLambda(), params.getC(), params.getNumIters(), w);
  }
  
  public void fitClassifier(List<LabeledVector> trainingSet, LoopType loopType,
      LearnerType learnerType, EtaType etaType, float lambda, float c,
      int numIters, MutableVector w) {
    switch (loopType) {
    case STOCHASTIC:
      stochasticOuterLoop(trainingSet, learnerType, etaType, lambda, c, numIters, w);
    case BALANCED_STOCHASTIC:
      balancedStochasticOuterLoop(trainingSet, learnerType, etaType, lambda, c, numIters, w);
    default:
      throw new IllegalArgumentException("Loop type " + loopType + " not supported.");
    }
  }

  // --------------------------------------------------- //
  //            Stochastic Loop Strategy Functions
  // --------------------------------------------------- //

  private LabeledVector randomExample(List<LabeledVector> data_set) {
    return data_set.get(rand.nextInt(data_set.size()));
  }
  
  void stochasticOuterLoop(List<LabeledVector> training_set,
      LearnerType learner_type,
      EtaType eta_type,
      float lambda,
      float c,
      int num_iters,
      MutableVector w) {
    for (int i = 1; i <= num_iters; ++i) {
      LabeledVector x = randomExample(training_set);
      float eta = getEta(eta_type, lambda, i);
      oneLearnerStep(learner_type, x, eta, c, lambda, w);
    }
  }  

  void balancedStochasticOuterLoop(List<LabeledVector> training_set,
      LearnerType learner_type,
      EtaType eta_type,
      float lambda,
      float c,
      int num_iters,
      MutableVector w) {
    // Create index of positives and negatives for fast sampling
    // of disagreeing pairs.
    List<Integer> positives = new ArrayList<Integer>();
    List<Integer> negatives = new ArrayList<Integer>();
    for (int i = 0; i < training_set.size(); ++i) {
      if (training_set.get(i).getLabel() > 0.0)
        positives.add(i);
      else
        negatives.add(i);
    }

    // For each iteration, randomly sample one positive and one negative and
    // take one gradient step for each.
    for (int i = 1; i <= num_iters; ++i) {
      float eta = getEta(eta_type, lambda, i);

      LabeledVector pos_x =
          training_set.get(positives.get(rand.nextInt(positives.size())));
      oneLearnerStep(learner_type, pos_x, eta, c, lambda, w);

      LabeledVector neg_x =
          training_set.get(negatives.get(rand.nextInt(negatives.size())));
      oneLearnerStep(learner_type, neg_x, eta, c, lambda, w);
    }
  }

  public void StochasticRocLoop(List<LabeledVector> training_set,
      LearnerType learner_type,
      EtaType eta_type,
      float lambda,
      float c,
      int num_iters,
      MutableVector w) {
    // Create index of positives and negatives for fast sampling
    // of disagreeing pairs.
    List<Integer> positives = new ArrayList<Integer>();
    List<Integer> negatives = new ArrayList<Integer>();
    for (int i = 0; i < training_set.size(); ++i) {
      if (training_set.get(i).getLabel() > 0.0)
        positives.add(i);
      else
        negatives.add(i);
    }

    // For each step, randomly sample one positive and one negative and
    // take a pairwise gradient step.
    for (int i = 1; i <= num_iters; ++i) {
      float eta = getEta(eta_type, lambda, i);
      LabeledVector pos_x =
          training_set.get(positives.get(rand.nextInt(positives.size())));
      LabeledVector neg_x =
          training_set.get(negatives.get(rand.nextInt(negatives.size())));
      oneLearnerRankStep(learner_type, pos_x, neg_x, eta, c, lambda, w);
    }
  }

  void stochasticClassificationAndRocLoop(List<LabeledVector> training_set,
      LearnerType learner_type,
      EtaType eta_type,
      float lambda,
      float c,
      float rank_step_probability,
      int num_iters,
      MutableVector w) {
    // Create index of positives and negatives for fast sampling
    // of disagreeing pairs.
    List<Integer> positives = new ArrayList<Integer>();
    List<Integer> negatives = new ArrayList<Integer>();
    for (int i = 0; i < training_set.size(); ++i) {
      if (training_set.get(i).getLabel() > 0.0)
        positives.add(i);
      else
        negatives.add(i);
    }

    for (int i = 1; i <= num_iters; ++i) {
      float eta = getEta(eta_type, lambda, i);
      if (rand.nextDouble() < rank_step_probability) {
        // For each step, randomly sample one positive and one negative and
        // take a pairwise gradient step.
        LabeledVector pos_x =
            training_set.get(positives.get(rand.nextInt(positives.size())));
        LabeledVector neg_x =
            training_set.get(negatives.get(rand.nextInt(negatives.size())));
        oneLearnerRankStep(learner_type, pos_x, neg_x, eta, c, lambda, w);
      } else {
        // Take a classification step.
        LabeledVector x = randomExample(training_set);
        oneLearnerStep(learner_type, x, eta, c, lambda, w);      
      }
    }
  }

  /*
  void stochasticClassificationAndRankLoop(List<LabeledVector> training_set,
      LearnerType learner_type,
      EtaType eta_type,
      float lambda,
      float c,
      float rank_step_probability,
      int num_iters,
      ClassifierVector w) {
    Map<String, Map<Float, List<Integer> > > group_id_y_to_index;
    Map<String, Integer> group_id_y_to_count;
    for (int i = 0; i < training_set.size(); ++i) {
      String group_id = training_set.get(i).GetGroupId();
      group_id_y_to_index[group_id][training_set.get(i).getLabel()].push_back(i);
      group_id_y_to_count[group_id] += 1;
    }

    for (int i = 1; i <= num_iters; ++i) {
      if (rand.nextFloat() < rank_step_probability) {
        // Take a rank step.
        LabeledVector a = RandomExample(training_set);
        String group_id = a.GetGroupId();
        float a_y = a.getLabel();
        const Map<Float, List<Integer>> y_to_list =
            group_id_y_to_index[group_id];
        int range = group_id_y_to_count[group_id] -
            group_id_y_to_index[group_id][a_y].size();
        if (range == 0) continue;
        int random_int = rand.nextInt(range);
        for (Map<Float, List<Integer> >::const_iterator iter =
            y_to_list.begin();
        iter != y_to_list.end();
        iter++) {
          if (iter.first == a_y) continue;
          if (random_int < iter.second.size()) {
            LabeledVector b = 
                training_set.get((iter.second)[random_int]);
            float eta = GetEta(eta_type, lambda, i);
            oneLearnerRankStep(learner_type, a, b, eta, c, lambda, w);
            break;
          }
          random_int -= iter.second.size();
        }
      } else {
        // Take a classification step.
        LabeledVector x = RandomExample(training_set);
        float eta = GetEta(eta_type, lambda, i);
        OneLearnerStep(learner_type, x, eta, c, lambda, w);
      }
    }
  }
   */

  /*
  void StochasticRankLoop(List<LabeledVector> training_set,
      LearnerType learner_type,
      EtaType eta_type,
      float lambda,
      float c,
      int num_iters,
      ClassifierVector w) {
    Map<String, Map<Float, List<Integer>>> group_id_y_to_index;
    Map<String, Integer> group_id_y_to_count;
    for (int i = 0; i < training_set.size(); ++i) {
      String group_id = training_set.get(i).GetGroupId();
      group_id_y_to_index[group_id][training_set.get(i).getLabel()].push_back(i);
      group_id_y_to_count[group_id] += 1;
    }

    for (int i = 1; i <= num_iters; ++i) {
      LabeledVector a = RandomExample(training_set);
      String group_id = a.GetGroupId();
      float a_y = a.getLabel();
      const Map<Float, List<Integer> >& y_to_list =
          group_id_y_to_index[group_id];
      int range =
          group_id_y_to_count[group_id] - group_id_y_to_index[group_id][a_y].size();
      if (range == 0) continue;
      int random_int = rand.nextInt(range);
      for (Map<Float, List<Integer> >::const_iterator iter = y_to_list.begin();
      iter != y_to_list.end();
      iter++) {
        if (iter.first == a_y) continue;
        if (random_int < iter.second.size()) {
          LabeledVector b = 
              training_set.get((iter.second)[random_int]);
          float eta = GetEta(eta_type, lambda, i);
          oneLearnerRankStep(learner_type, a, b, eta, c, lambda, w);
          break;
        }
        random_int -= iter.second.size();
      }
    }
  }
   */
  /*
  void StochasticQueryNormRankLoop(List<LabeledVector> training_set,
      LearnerType learner_type,
      EtaType eta_type,
      float lambda,
      float c,
      int num_iters,
      ClassifierVector w) {
    // Create a map of group id's to examples.
    Map<String, List<Integer>> group_id_to_examples;
    for (int i = 0; i < training_set.size(); ++i) {
      String group_id = training_set.get(i).GetGroupId();
      group_id_to_examples.get(group_id).add(i);
    }

    // Assign each group id a unique integer id.
    Map<Integer, List<Integer> > group_id_index;
    int i = 0;
    for (Map<String, List<Integer> >::iterator group_id_iter =
        group_id_to_examples.begin();
    group_id_iter != group_id_to_examples.end();
    ++group_id_iter) {
      group_id_index[i] = &(group_id_iter.second);
      ++i;
    }

    for (int i = 1; i <= num_iters; ++i) {
      int group_id = rand.nextInt(group_id_index.size());
      List<Integer> group_index = group_id_index.get(group_id);
      int group_index_size = group_index.size();
      LabeledVector a =
          training_set.get(group_index.get(rand.nextInt(group_index_size)));

      boolean found_differing_pair = false;
      int num_chances = 1000;
      while (!found_differing_pair && --num_chances > 0) {
        LabeledVector b =
            training_set.get(group_index.get(rand.nextInt(group_index_size)));

        found_differing_pair = a.getLabel() != b.getLabel();
        if (found_differing_pair) {
          float eta = GetEta(eta_type, lambda, i);
          oneLearnerRankStep(learner_type, a, b, eta, c, lambda, w);
          break;
        }
      }
    } 
  }
   */

  // --------------------------------------------------- //
  //       single Stochastic Step Strategy Methods
  // --------------------------------------------------- //

  boolean oneLearnerStep(LearnerType learnerType,
      LabeledVector x,
      float eta,
      float c,
      float lambda,
      MutableVector w) {
    switch (learnerType) {
    case PEGASOS:
      return singlePegasosStep(x, eta, lambda, w);
//    case MARGIN_PERCEPTRON:
//      return singleMarginPerceptronStep(x, eta, c, w);
    case PASSIVE_AGGRESSIVE:
      return singlePassiveAggressiveStep(x, lambda, c, w);
    case LOGREG_PEGASOS:
      return singlePegasosLogRegStep(x, eta, lambda, w);
    case LOGREG:
      return singleLogRegStep(x, eta, lambda, w);
    case LMS_REGRESSION:
      return singleLeastMeanSquaresStep(x, eta, lambda, w);
    case SGD_SVM:
      return singleSgdSvmStep(x, eta, lambda, w);
    case ROMMA:
      return singleRommaStep(x, w);
    default:
      throw new IllegalArgumentException("Error: learner_type " + learnerType
          + " not supported.");
    }
  }

  boolean oneLearnerRankStep(LearnerType learner_type,
      LabeledVector a,
      LabeledVector b,
      float eta,
      float c,
      float lambda,
      MutableVector w) {
    switch (learner_type) {
    case PEGASOS:
      return singlePegasosRankStep(a, b, eta, lambda, w);
//    case MARGIN_PERCEPTRON:
//      return singleMarginPerceptronRankStep(a, b, eta, c, w);
      //    case PASSIVE_AGGRESSIVE:
      //      return singlePassiveAggressiveRankStep(a, b, lambda, c, w);
    case LOGREG_PEGASOS:
      return singlePegasosLogRegRankStep(a, b, eta, lambda, w);
    case LOGREG:
      return singleLogRegRankStep(a, b, eta, lambda, w);
    case LMS_REGRESSION:
      return singleLeastMeanSquaresRankStep(a, b, eta, lambda, w);
    case SGD_SVM:
      return singleSgdSvmRankStep(a, b, eta, lambda, w);
      //    case ROMMA:
      //      return singleRommaRankStep(a, b, w);
    default:
      throw new IllegalArgumentException("Error: learner_type " + learner_type
          + " not supported.");
    }
  }

  // --------------------------------------------------- //
  //            single Stochastic Step Functions
  // --------------------------------------------------- //

  boolean singlePegasosStep(LabeledVector x,
      float eta,
      float lambda,
      MutableVector w) {
    float p = x.getLabel() * w.innerProduct(x);

    regularizeL2(eta, lambda, w);
    // If x has non-zero loss, perform gradient step in direction of x.
    if (p < 1.0 && x.getLabel() != 0.0) {
      w.addVector(x, (eta * x.getLabel())); 
    }

    pegasosProjection(lambda, w);
    return (p < 1.0 && x.getLabel() != 0.0);
  }

  boolean singleRommaStep(LabeledVector x,
      MutableVector w) {
    float wx = w.innerProduct(x);
    float p = x.getLabel() * wx;
    final float kVerySmallNumber = 0.0000000001f;

    // If x has non-zero loss, perform gradient step in direction of x.
    if (p < 1.0 && x.getLabel() != 0.0) {
      float xx = x.getSquaredNorm();
      float ww = w.getSquaredNorm();
      float c = ((xx * ww) - p + kVerySmallNumber) /
          ((xx * ww) - (wx * wx) + kVerySmallNumber);

      float d = (ww * (x.getLabel() - wx) + kVerySmallNumber) /
          ((xx * ww) - (wx * wx) + kVerySmallNumber);

      // Avoid numerical problems caused by examples of extremely low magnitude.
      if (c >= 0.0) {
        w.scaleBy(c);
        w.addVector(x, d); 
      }
    }

    return (p < 1.0 && x.getLabel() != 0.0);
  }

  boolean singleSgdSvmStep(LabeledVector x,
      float eta,
      float lambda,
      MutableVector w) {
    float p = x.getLabel() * w.innerProduct(x);    

    regularizeL2(eta, lambda, w);    
    // If x has non-zero loss, perform gradient step in direction of x.
    if (p < 1.0 && x.getLabel() != 0.0) {
      w.addVector(x, (eta * x.getLabel())); 
    }

    return (p < 1.0 && x.getLabel() != 0.0);
  }

  boolean singleMarginPerceptronStep(LabeledVector x,
      float eta,
      float c,
      MutableVector w) {
    if (x.getLabel() * w.innerProduct(x) <= c) {
      w.addVector(x, eta * x.getLabel());
      return true;
    } else {
      return false;
    }
  }

  boolean singlePegasosLogRegStep(LabeledVector x,
      float eta,
      float lambda,
      MutableVector w) {
    float loss = x.getLabel() / (float)(1 + Math.exp(x.getLabel() * w.innerProduct(x)));

    regularizeL2(eta, lambda, w);    
    w.addVector(x, (eta * loss));
    pegasosProjection(lambda, w);
    return (true);
  }

  boolean singleLogRegStep(LabeledVector x,
      float eta,
      float lambda,
      MutableVector w) {
    float loss = x.getLabel() / (float)(1 + Math.exp(x.getLabel() * w.innerProduct(x)));

    regularizeL2(eta, lambda, w);    
    w.addVector(x, (eta * loss));
    return (true);
  }

  boolean singleLeastMeanSquaresStep(LabeledVector x,
      float eta,
      float lambda,
      MutableVector w) {
    float loss = x.getLabel() - w.innerProduct(x);
    regularizeL2(eta, lambda, w);    
    w.addVector(x, (eta * loss));
    pegasosProjection(lambda, w);
    return (true);
  }

  boolean singlePassiveAggressiveStep(LabeledVector x,
      float lambda,
      float max_step,
      MutableVector w) {
    float p = 1 - (x.getLabel() * w.innerProduct(x));    
    // If x has non-zero loss, perform gradient step in direction of x.
    if (p > 0.0 && x.getLabel() != 0.0) {
      float step = p / (float)x.getSquaredNorm();
      if (step > max_step) step = max_step;
      w.addVector(x, (step * x.getLabel())); 
    }

    if (lambda > 0.0) {
      pegasosProjection(lambda, w);
    }
    return (p < 1.0 && x.getLabel() != 0.0);
  }

  /*
  boolean singlePassiveAggressiveRankStep(LabeledVector a,
      LabeledVector b,
      float lambda,
      float max_step,
      ClassifierVector w) {
    float y = (a.getLabel() > b.getLabel()) ? 1.0f :
      (a.getLabel() < b.getLabel()) ? -1.0f : 0.0f;
    float p = 1 - (y * w.innerProductOnDifference(a, b)); 
    // If (a-b) has non-zero loss, perform gradient step in direction of x.
    if (p > 0.0 && y != 0.0) {
      // Compute squared norm of (a-b).
      int i = 0;
      int j = 0;
      float squared_norm = 0;
      while (i < a.NumFeatures() || j < b.NumFeatures()) {
        int a_feature = (i < a.NumFeatures()) ? a.FeatureAt(i) : Integer.MAX_VALUE;
        int b_feature = (j < b.NumFeatures()) ? b.FeatureAt(j) : Integer.MAX_VALUE;
        if (a_feature < b_feature) {
          squared_norm += a.ValueAt(i) * a.ValueAt(i);
          ++i;
        } else if (b_feature < a_feature) {
          squared_norm += b.ValueAt(j) * b.ValueAt(j);
          ++j;
        } else {
          squared_norm += (a.ValueAt(i) - b.ValueAt(j)) * (a.ValueAt(i) - b.ValueAt(j));
          ++i;
          ++j;
        }
      }
      float step = p / squared_norm;
      if (step > max_step) step = max_step;
      w.AddVector(a, (step * y)); 
      w.AddVector(b, (step * y * -1.0f)); 
    }

    if (lambda > 0.0) {
      PegasosProjection(lambda, w);
    }
    return (p > 0 && y != 0.0);
  }
   */

  boolean singlePegasosRankStep(LabeledVector a,
      LabeledVector b,
      float eta,
      float lambda,
      MutableVector w) {
    float y = (a.getLabel() > b.getLabel()) ? 1.0f :
      (a.getLabel() < b.getLabel()) ? -1.0f : 0.0f;
    float p = y * w.innerProductOnDifference(a, b);

    regularizeL2(eta, lambda, w);

    // If (a - b) has non-zero loss, perform gradient step.         
    if (p < 1.0 && y != 0.0) {
      w.addVector(a, (eta * y));
      w.addVector(b, (-1.0f * eta * y));
    }

    pegasosProjection(lambda, w);
    return (p < 1.0 && y != 0.0);
  }

  boolean singleSgdSvmRankStep(LabeledVector a,
      LabeledVector b,
      float eta,
      float lambda,
      MutableVector w) {
    float y = (a.getLabel() > b.getLabel()) ? 1.0f :
      (a.getLabel() < b.getLabel()) ? -1.0f : 0.0f;
    float p = y * w.innerProductOnDifference(a, b);

    regularizeL2(eta, lambda, w);

    // If (a - b) has non-zero loss, perform gradient step.         
    if (p < 1.0 && y != 0.0) {
      w.addVector(a, (eta * y));
      w.addVector(b, (-1.0f * eta * y));
    }

    return (p < 1.0 && y != 0.0);
  }

  boolean singleLeastMeanSquaresRankStep(LabeledVector a,
      LabeledVector b,
      float eta,
      float lambda,
      MutableVector w) {
    float y = (a.getLabel() - b.getLabel());
    float loss = y - w.innerProductOnDifference(a, b);

    regularizeL2(eta, lambda, w);
    w.addVector(a, (eta * loss));
    w.addVector(b, (-1.0f * eta * loss));
    pegasosProjection(lambda, w);
    return (true);
  }

  /*
  boolean singleRommaRankStep(LabeledVector a,
      LabeledVector b,
      ClassifierVector w) {
    // Not the most efficient approach, but it takes care of
    // computing the squared norm of x with minimal coding effort.
    float y = (a.getLabel() > b.getLabel()) ? 1.0f :
      (a.getLabel() < b.getLabel()) ? -1.0f : 0.0f;
    LabeledVector x_diff = new LabeledVector(a, b, y);
    if (y != 0.0) {
      return singleRommaStep(x_diff, w);
    } else {
      return false;
    }
  }
   */

  boolean singlePegasosLogRegRankStep(LabeledVector a,
      LabeledVector b,
      float eta,
      float lambda,
      MutableVector w) {
    float y = (a.getLabel() > b.getLabel()) ? 1.0f :
      (a.getLabel() < b.getLabel()) ? -1.0f : 0.0f;
    float loss = y / (1 + (float)Math.exp(y * w.innerProductOnDifference(a, b)));
    regularizeL2(eta, lambda, w);    

    w.addVector(a, (eta * loss));
    w.addVector(b, (-1.0f * eta * loss));

    pegasosProjection(lambda, w);
    return (true);
  }

  boolean singleLogRegRankStep(LabeledVector a,
      LabeledVector b,
      float eta,
      float lambda,
      MutableVector w) {
    float y = (a.getLabel() > b.getLabel()) ? 1.0f :
      (a.getLabel() < b.getLabel()) ? -1.0f : 0.0f;
    float loss = y / (1 + (float)Math.exp(y * w.innerProductOnDifference(a, b)));
    regularizeL2(eta, lambda, w);    

    w.addVector(a, (eta * loss));
    w.addVector(b, (-1.0f * eta * loss));
    return (true);
  }

  boolean singleMarginPerceptronRankStep(LabeledVector a,
      LabeledVector b,
      float eta,
      float c,
      MutableVector w) {
    float y = (a.getLabel() > b.getLabel()) ? 1.0f :
      (a.getLabel() < b.getLabel()) ? -1.0f : 0.0f;
    if (y * w.innerProductOnDifference(a, b) <= c) {
      w.addVector(a, eta);
      w.addVector(b, -1.0f * eta);
      return true;
    } else {
      return false;
    }
  }

  boolean singlePegasosRankWithTiesStep(LabeledVector rank_a,
      LabeledVector rank_b,
      LabeledVector tied_a,
      LabeledVector tied_b,
      float eta,
      float lambda,
      MutableVector w) {
    float rank_y = (rank_a.getLabel() > rank_b.getLabel()) ? 1.0f :
      (rank_a.getLabel() < rank_b.getLabel()) ? -1.0f : 0.0f;
    float rank_p = rank_y * w.innerProductOnDifference(rank_a, rank_b);
    float tied_p = w.innerProductOnDifference(tied_a, tied_b);

    regularizeL2(eta, lambda, w);

    // If (rank_a - rank_b) has non-zero loss, perform gradient step.         
    if (rank_p < 1.0 && rank_y != 0.0) {
      w.addVector(rank_a, (eta * rank_y));
      w.addVector(rank_b, (-1.0f * eta * rank_y));
    }

    // The value of tied_p should ideally be 0.0.  We penalize with squared
    // loss for predictions away from 0.0.
    if (tied_a.getLabel() == tied_b.getLabel()) {
      w.addVector(tied_a, (eta * (0.0f - tied_p)));
      w.addVector(tied_b, (-1.0f * eta * (0.0f - tied_p)));
    }

    pegasosProjection(lambda, w);
    return (true);
  }

  void regularizeL2(float eta, float lambda, MutableVector w) {
    float scaling_factor = 1.0f - (eta * lambda);
    if (scaling_factor > MIN_SCALING_FACTOR) {
      w.scaleBy(1.0 - (eta * lambda));  
    } else {
      w.scaleBy(MIN_SCALING_FACTOR); 
    }
  }

  void L2RegularizeSeveralSteps(float eta,
      float lambda,
      float effective_steps,
      MutableVector w) {
    float scaling_factor = 1.0f - (eta * lambda);
    scaling_factor = (float)Math.pow(scaling_factor, effective_steps);
    if (scaling_factor > MIN_SCALING_FACTOR) {
      w.scaleBy(1.0 - (eta * lambda));  
    } else {
      w.scaleBy(MIN_SCALING_FACTOR); 
    }
  }

  void pegasosProjection(float lambda, MutableVector w) {
    float projection_val = 1 / (float)Math.sqrt(lambda * w.getSquaredNorm());
    if (projection_val < 1.0) {
      w.scaleBy(projection_val);
    }
  }
  
  private float getEta (EtaType eta_type, float lambda, int i) {
    switch (eta_type) {
    case BASIC_ETA:
      return 10.0f / (i + 10.0f);
    case PEGASOS_ETA:
      return 1.0f / (lambda * i);
    case CONSTANT:
      return 0.02f;
    default:
      throw new IllegalArgumentException("EtaType " + eta_type + " not supported.");
    }
  }

}