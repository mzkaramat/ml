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

package com.cloudera.science.ml.client.cmd;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.crunch.PCollection;
import org.apache.crunch.Pair;
import org.apache.crunch.Pipeline;
import org.apache.crunch.materialize.pobject.CollectionPObject;
import org.apache.hadoop.conf.Configuration;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.beust.jcommander.ParametersDelegate;
import com.beust.jcommander.internal.Lists;
import com.cloudera.science.ml.classifier.core.EtaUpdate;
import com.cloudera.science.ml.classifier.core.OnlineLearnerParams;
import com.cloudera.science.ml.classifier.core.OnlineLearnerRun;
import com.cloudera.science.ml.classifier.core.OnlineLearnerRuns;
import com.cloudera.science.ml.classifier.parallel.FitFn;
import com.cloudera.science.ml.classifier.parallel.ParallelLearner;
import com.cloudera.science.ml.classifier.parallel.SimpleFitFn;
import com.cloudera.science.ml.classifier.parallel.types.ClassifierAvros;
import com.cloudera.science.ml.classifier.simple.LinRegOnlineLearner;
import com.cloudera.science.ml.classifier.simple.LogRegOnlineLearner;
import com.cloudera.science.ml.classifier.simple.SimpleOnlineLearner;
import com.cloudera.science.ml.client.params.PipelineParameters;
import com.cloudera.science.ml.client.params.VectorInputParameters;
import com.cloudera.science.ml.client.util.AvroIO;
import com.cloudera.science.ml.core.vectors.LabeledVector;
import com.cloudera.science.ml.parallel.crossfold.CrossfoldFn;
import com.cloudera.science.ml.parallel.distribute.DistributeFn;
import com.cloudera.science.ml.parallel.distribute.SimpleDistributeFn;
import com.cloudera.science.ml.parallel.fn.ShuffleFn;
import com.cloudera.science.ml.classifier.avro.MLOnlineLearnerRuns;

@Parameters(commandDescription = "Fits a set of classification models to a labeled dataset")
public class FitCommand implements Command {
  
  private static final int LAMBDA_POWER_RANGE_BOTTOM = -6;
  private static final int LAMBDA_POWER_RANGE_TOP = 6;
  
  @Parameter(names = "--learner-type",
      description = "The kind of classifier to train, such as linreg or logreg")
  private String learnerType;
  
  @Parameter(names = "--eta-type",
      description = "The eta update to use in the SGD, either CONSTANT, BASIC, or PEGASOS")
  private String etaType = "CONSTANT";

  @Parameter(names = "--num-lambdas",
      description = "The number of different regularization parameters to try")
  private int numLambdas;
  
  @Parameter(names = "--num-crossfolds",
    description = "The number of cross validation subsets")
  private int numCrossfolds;
  
  @Parameter(names = "--num-partitions",
    description = "The number of partitions to split each training fold into")
  private int numPartitions = 1;
  
  @Parameter(names = "--regularization-type",
    description = "The type of regularization to perform, either L1 or L2")
  private String regularizationType = "L2";

  @Parameter(names = "--seed",
      description = "Seed for the random number generators")
  private long seed;
  
  @Parameter(names = "--output-file", required=true,
      description = "A local file to write the output to (as Avro OnlineLearnerRuns records)")
  private String outputFile;

  @ParametersDelegate
  private PipelineParameters pipelineParams = new PipelineParameters();
  
  @ParametersDelegate
  private VectorInputParameters inputParams = new VectorInputParameters();
  
  @Override
  public int execute(Configuration conf) throws IOException {
    Pipeline p = pipelineParams.create(FitCommand.class, conf);

    PCollection<LabeledVector> labeledVectors = inputParams.getLabeledVectors(p);
    
    // Generate learners with an array of regularization parameters
    OnlineLearnerParams.Builder paramsBuilder = OnlineLearnerParams.builder()
        .etaUpdate(parseEtaUpdate(etaType));

    float[] lambdas = lambdas(numLambdas);
    List<SimpleOnlineLearner> learners = new ArrayList<SimpleOnlineLearner>();
    for (float lambda : lambdas) {
      // TODO: regularization type
      OnlineLearnerParams params = paramsBuilder.L2(lambda).build();
      learners.add(makeLearner(params));
    }
    FitFn fitFn = new SimpleFitFn(learners);
    
    ShuffleFn<LabeledVector> shuffleFn = new ShuffleFn<LabeledVector>(seed);
    // TODO: use LabelSeparatingShuffleFn if loop type is ranked / balanced?
    
    CrossfoldFn<Pair<Integer, LabeledVector>> crossfoldFn =
        new CrossfoldFn<Pair<Integer, LabeledVector>>(numCrossfolds, seed);
    
    DistributeFn<Integer, Pair<Integer, LabeledVector>> distributeFn =
        new SimpleDistributeFn<Integer, Pair<Integer, LabeledVector>>(
            numPartitions, seed);
    
    ParallelLearner learner = new ParallelLearner();
    PCollection<OnlineLearnerRun> pruns =
        learner.runPipeline(labeledVectors, shuffleFn, crossfoldFn, distributeFn, fitFn);
    
    // Pull down results
    Collection<OnlineLearnerRun> runs =
        new CollectionPObject<OnlineLearnerRun>(pruns).getValue();
    
    // Write them out to local file, along with metadata
    OnlineLearnerRuns runsAndMetadata = new OnlineLearnerRuns(runs, seed, numCrossfolds);
    AvroIO.write(Lists.newArrayList(
        ClassifierAvros.fromOnlineLearnerRuns(runsAndMetadata)), new File(outputFile));
    p.done();
    
    return 0;
  }
  
  private SimpleOnlineLearner makeLearner(OnlineLearnerParams params) {
    if (learnerType.equalsIgnoreCase("logreg")) {
      return new LogRegOnlineLearner(params);
    } else if (learnerType.equalsIgnoreCase("linreg")) {
      return new LinRegOnlineLearner(params);
    } else {
      throw new IllegalArgumentException("Invalid learner type: " + learnerType);
    }
  }
  
  private EtaUpdate parseEtaUpdate(String etaUpdate) {
    if (etaUpdate.equalsIgnoreCase("CONSTANT")) {
      return EtaUpdate.CONSTANT;
    } else if (etaUpdate.equalsIgnoreCase("BASIC")) {
      return EtaUpdate.BASIC_ETA;
    } else if (etaUpdate.equalsIgnoreCase("PEGASOS")) {
      return EtaUpdate.PEGASOS_ETA;
    } else {
      throw new IllegalArgumentException("Invalid eta update: " + etaUpdate);
    }
  }
  
  private float[] lambdas(int numLambdas) {
    float[] lambdas = new float[numLambdas];
    for (int i = 0; i < lambdas.length; i++) {
      int power = (LAMBDA_POWER_RANGE_TOP - LAMBDA_POWER_RANGE_BOTTOM) * i / numLambdas
          + LAMBDA_POWER_RANGE_BOTTOM;
      lambdas[i] = (float)Math.pow(2.0, power);
    }
    return lambdas;
  }
  
  @Override
  public String getDescription() {
    return "Fits a set of classification models to a labeled dataset";
  }
}
