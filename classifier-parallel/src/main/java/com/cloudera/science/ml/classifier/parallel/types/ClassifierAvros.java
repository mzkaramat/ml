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

package com.cloudera.science.ml.classifier.parallel.types;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.crunch.MapFn;
import org.apache.crunch.types.avro.AvroType;
import org.apache.crunch.types.avro.Avros;

import com.cloudera.science.ml.classifier.core.Classifier;
import com.cloudera.science.ml.classifier.core.ModelScore;
import com.cloudera.science.ml.classifier.core.OnlineLearnerRun;
import com.cloudera.science.ml.classifier.core.EtaUpdate;
import com.cloudera.science.ml.classifier.core.OnlineLearnerParams;
import com.cloudera.science.ml.classifier.core.OnlineLearnerRuns;
import com.cloudera.science.ml.classifier.core.WeightVector;
import com.cloudera.science.ml.classifier.avro.MLClassifier;
import com.cloudera.science.ml.classifier.avro.MLModelScore;
import com.cloudera.science.ml.classifier.avro.MLOnlineLearnerRun;
import com.cloudera.science.ml.classifier.avro.MLOnlineLearnerRuns;
import com.cloudera.science.ml.classifier.avro.MLOnlineLearnerParams;
import com.cloudera.science.ml.core.vectors.VectorConvert;
import com.cloudera.science.ml.avro.MLVector;

/**
 * Factory methods for creating {@code PType} instances for use with the ML Parallel libraries.
 */
public class ClassifierAvros {
  public static AvroType<OnlineLearnerParams> params() {
    return params;
  }
  
  public static AvroType<Classifier> classifier() {
    return classifier;
  }
  
  public static AvroType<OnlineLearnerRun> onlineLearnerRun() {
    return onlineLearnerRun;
  }
  
  public static AvroType<OnlineLearnerRuns> onlineLearnerRuns() {
    return onlineLearnerRuns;
  }
  
  public static AvroType<ModelScore> modelScore() {
    return modelScore;
  }
  
  private static final AvroType<OnlineLearnerParams> params = Avros.derived(OnlineLearnerParams.class,
      new MapFn<MLOnlineLearnerParams, OnlineLearnerParams>() {
        @Override
        public OnlineLearnerParams map(MLOnlineLearnerParams mlParams) {
          return toParams(mlParams);
        }
      },
      new MapFn<OnlineLearnerParams, MLOnlineLearnerParams>() {
        @Override
        public MLOnlineLearnerParams map(OnlineLearnerParams params) {
          return fromParams(params);
        }
      },
      Avros.specifics(MLOnlineLearnerParams.class));
  
  private static final AvroType<Classifier> classifier = Avros.derived(Classifier.class,
      new MapFn<MLClassifier, Classifier>() {
        @Override
        public Classifier map(MLClassifier mlClassifier) {
          return toClassifier(mlClassifier);
        }
      },
      new MapFn<Classifier, MLClassifier>() {
        @Override
        public MLClassifier map(Classifier classifier) {
          return fromClassifier(classifier);
        }
      },
      Avros.specifics(MLClassifier.class));
  
  private static final AvroType<OnlineLearnerRun> onlineLearnerRun = Avros.derived(OnlineLearnerRun.class,
      new MapFn<MLOnlineLearnerRun, OnlineLearnerRun>() {
        @Override
        public OnlineLearnerRun map(MLOnlineLearnerRun mlOnlineLearnerRun) {
          return toOnlineLearnerRun(mlOnlineLearnerRun);
        }
      },
      new MapFn<OnlineLearnerRun, MLOnlineLearnerRun>() {
        @Override
        public MLOnlineLearnerRun map(OnlineLearnerRun onlineLearnerRun) {
          return fromOnlineLearnerRun(onlineLearnerRun);
        }
      },
      Avros.specifics(MLOnlineLearnerRun.class));
  
  private static final AvroType<OnlineLearnerRuns> onlineLearnerRuns = Avros.derived(OnlineLearnerRuns.class,
      new MapFn<MLOnlineLearnerRuns, OnlineLearnerRuns>() {
        @Override
        public OnlineLearnerRuns map(MLOnlineLearnerRuns mlOnlineLearnerRuns) {
          return toOnlineLearnerRuns(mlOnlineLearnerRuns);
        }
      },
      new MapFn<OnlineLearnerRuns, MLOnlineLearnerRuns>() {
        @Override
        public MLOnlineLearnerRuns map(OnlineLearnerRuns onlineLearnerRuns) {
          return fromOnlineLearnerRuns(onlineLearnerRuns);
        }
      },
      Avros.specifics(MLOnlineLearnerRuns.class));
  
  private static final AvroType<ModelScore> modelScore = Avros.derived(ModelScore.class,
      new MapFn<MLModelScore, ModelScore>() {
        @Override
        public ModelScore map(MLModelScore mlModelScore) {
          return toModelScore(mlModelScore);
        }
      },
      new MapFn<ModelScore, MLModelScore>() {
        @Override
        public MLModelScore map(ModelScore ModelScore) {
          return fromModelScore(ModelScore);
        }
      },
      Avros.specifics(MLModelScore.class));
  
  private static OnlineLearnerParams toParams(MLOnlineLearnerParams mlParams) {
    String etaUpdateClass = mlParams.getEtaUpdate().toString();
    EtaUpdate etaUpdate;
    try {
      etaUpdate = (EtaUpdate)Class.forName(etaUpdateClass).newInstance();
    } catch (Exception ex) {
      throw new RuntimeException("Error instantiating eta update", ex);
    }
    return new OnlineLearnerParams.Builder()
      .L2(mlParams.getLambda())
      .etaUpdate(etaUpdate)
      .pegasos(mlParams.getPegasos())
      .L1(mlParams.getL1Radius(), mlParams.getL1Iterations())
      .build();
  }
  
  private static MLOnlineLearnerParams fromParams(OnlineLearnerParams params) {
    return MLOnlineLearnerParams.newBuilder()
        .setLambda(params.lambda())
        .setEtaUpdate(params.etaUpdate().getClass().getName())
        .setPegasos(params.pegasos())
        .setL1Radius(params.l1Radius())
        .setL1Iterations(params.l1Iterations())
        .build();
  }
  
  private static Classifier toClassifier(MLClassifier mlClassifier) {
    String classifierClass = mlClassifier.getClassName().toString();
    WeightVector weights = new WeightVector(VectorConvert.toVector(mlClassifier.getWeights()));
    Classifier classifier;
    try {
      return (Classifier)Class.forName(classifierClass)
          .getDeclaredConstructor(WeightVector.class).newInstance(weights);
    } catch (Exception ex) {
      throw new RuntimeException("Error instantiating classifier", ex);
    }
  }
  
  private static MLClassifier fromClassifier(Classifier classifier) {
    return MLClassifier.newBuilder()
        .setWeights(VectorConvert.fromVector(classifier.getWeights().toVector()))
        .setClassName(classifier.getClass().getName())
        .build();
  }
  
  private static OnlineLearnerRun toOnlineLearnerRun(MLOnlineLearnerRun mlRun) {
    return new OnlineLearnerRun(toClassifier(mlRun.getClassifier()),
        toParams(mlRun.getParams()), mlRun.getFold(), mlRun.getPartition());
  }
  
  private static MLOnlineLearnerRun fromOnlineLearnerRun(OnlineLearnerRun run) {
    return MLOnlineLearnerRun.newBuilder()
        .setClassifier(fromClassifier(run.getClassifier()))
        .setParams(fromParams(run.getParams()))
        .setFold(run.getFold())
        .setPartition(run.getPartition())
        .build();
  }
  
  private static OnlineLearnerRuns toOnlineLearnerRuns(MLOnlineLearnerRuns mlRuns) {
    List<MLOnlineLearnerRun> avroRuns = mlRuns.getRuns();
    List<OnlineLearnerRun> runs = new ArrayList<OnlineLearnerRun>(avroRuns.size());
    for (MLOnlineLearnerRun avroRun : avroRuns) {
      runs.add(toOnlineLearnerRun(avroRun));
    }
    return new OnlineLearnerRuns(runs, mlRuns.getCrossfoldSeed(), mlRuns.getNumFolds());
  }
  
  public static MLOnlineLearnerRuns fromOnlineLearnerRuns(OnlineLearnerRuns runs) {
    Collection<OnlineLearnerRun> runsList = runs.getRuns();
    List<MLOnlineLearnerRun> avroRuns = new ArrayList<MLOnlineLearnerRun>(runsList.size());
    for (OnlineLearnerRun run : runsList) {
      avroRuns.add(fromOnlineLearnerRun(run));
    }

    return MLOnlineLearnerRuns.newBuilder().setCrossfoldSeed(runs.getSeed())
        .setNumFolds(runs.getNumFolds()).setRuns(avroRuns).build();
  }
  
  public static ModelScore toModelScore(MLModelScore mlModelScore) {
    return new ModelScore(mlModelScore.getTrueNegatives(),
        mlModelScore.getFalseNegatives(),
        mlModelScore.getTruePositives(),
        mlModelScore.getFalsePositives());
  }
  
  public static MLModelScore fromModelScore(ModelScore modelScore) {
    return MLModelScore.newBuilder().setTrueNegatives(modelScore.getTrueNegatives())
        .setFalseNegatives(modelScore.getFalseNegatives())
        .setTruePositives(modelScore.getTruePositives())
        .setFalsePositives(modelScore.getFalsePositives())
        .build();
  }
}
