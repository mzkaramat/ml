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
import com.cloudera.science.ml.classifier.parallel.BalancedFitFn;
import com.cloudera.science.ml.classifier.parallel.FitFn;
import com.cloudera.science.ml.classifier.parallel.ParallelLearner;
import com.cloudera.science.ml.classifier.parallel.RocFitFn;
import com.cloudera.science.ml.classifier.parallel.SimpleFitFn;
import com.cloudera.science.ml.classifier.parallel.types.ClassifierAvros;
import com.cloudera.science.ml.classifier.rank.RankOnlineLearner;
import com.cloudera.science.ml.classifier.simple.SimpleOnlineLearner;
import com.cloudera.science.ml.client.params.PipelineParameters;
import com.cloudera.science.ml.client.params.VectorInputParameters;
import com.cloudera.science.ml.client.util.AvroIO;
import com.cloudera.science.ml.client.util.ParamUtils;
import com.cloudera.science.ml.client.util.ParameterInterpolation;
import com.cloudera.science.ml.core.vectors.LabeledVector;
import com.cloudera.science.ml.parallel.crossfold.CrossfoldFn;
import com.cloudera.science.ml.parallel.distribute.DistributeFn;
import com.cloudera.science.ml.parallel.distribute.SimpleDistributeFn;
import com.cloudera.science.ml.parallel.fn.LabelSeparatingShuffleFn;
import com.cloudera.science.ml.parallel.fn.ShuffleFn;

@Parameters(commandDescription = "Fits a set of classification models to a labeled dataset")
public class FitCommand implements Command {
  private static final int DEFAULT_NUM_LAMBDAS = 4;
  private static final ParameterInterpolation DEFAULT_LAMBDA_INTERPOLATION =
      ParameterInterpolation.EXPONENTIAL;
  private static final float DEFAULT_LAMBDA_BOTTOM = .5f;
  private static final float DEFAULT_LAMBDA_TOP = 4.0f;
  
  @Parameter(names = "--loop-types",
      description = "The strategy for looping through the data, either simple, balanced, or rank")
  private String loopType = "simple";
  
  @Parameter(names = "--rarer-label",
      description = "For balanced and rank loops, the rarer label, of whose vectors will be held in memory")
  private double rarerLabel = 1.0;
  
  @Parameter(names = "--learner-types",
      description = "The kind of classifier to train, such as linreg or logreg")
  private String learnerType;
  
  @Parameter(names = "--eta-type",
      description = "The eta update to use in the SGD, either CONSTANT, BASIC, or PEGASOS")
  private String etaType = "CONSTANT";

  @Parameter(names = "--lambdas",
      description = "The regularization parameters to try, formatted like " +
      		" or \".5-2.0,3,lin\"")
  private String lambdas;
  
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
    
    if (!(regularizationType.equalsIgnoreCase("L1") ||
        regularizationType.equalsIgnoreCase("L2"))) {
      throw new IllegalArgumentException("Invalid regularization type: "
        + regularizationType);
    }
    boolean l2 = regularizationType.equalsIgnoreCase("L2") ? true : false;
    
    float[] lambdaVals = ParamUtils.parseMultivaluedParameter(lambdas,
        DEFAULT_LAMBDA_BOTTOM, DEFAULT_LAMBDA_TOP, DEFAULT_NUM_LAMBDAS,
        DEFAULT_LAMBDA_INTERPOLATION);
    List<OnlineLearnerParams> paramsList = makeParams(ParamUtils.parseEtaUpdates(etaType),
        lambdaVals, l2);
    
    ShuffleFn<LabeledVector> shuffleFn;
    FitFn fitFn;
    if (loopType.equalsIgnoreCase("simple")) {
      shuffleFn = new ShuffleFn<LabeledVector>(seed);
      fitFn = new SimpleFitFn(makeLearners(paramsList));
    } else if (loopType.equalsIgnoreCase("balanced")) {
      shuffleFn = new LabelSeparatingShuffleFn(seed, rarerLabel);
      fitFn = new BalancedFitFn(makeLearners(paramsList));
    } else if (loopType.equals("rank")) {
      shuffleFn = new LabelSeparatingShuffleFn(seed, rarerLabel);
      fitFn = new RocFitFn(makeRankLearners(paramsList));
    } else {
      throw new IllegalArgumentException("Illegal loopType: " + loopType);
    }

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
  
  private List<SimpleOnlineLearner> makeLearners(List<OnlineLearnerParams> params) {
    List<SimpleOnlineLearner> learners = new ArrayList<SimpleOnlineLearner>();
    for (OnlineLearnerParams learnerParams : params) {
      learners.addAll(ParamUtils.makeSimpleLearners(learnerParams, learnerType));
    }
    return learners;
  }
  
  private List<RankOnlineLearner> makeRankLearners(List<OnlineLearnerParams> params) {
    List<RankOnlineLearner> learners = new ArrayList<RankOnlineLearner>();
    for (OnlineLearnerParams learnerParams : params) {
      learners.addAll(ParamUtils.makeRankLearners(learnerParams, learnerType));
    }
    return learners;
  }
  
  private List<OnlineLearnerParams> makeParams(EtaUpdate[] etaUpdates,
      float[] lambdas, boolean l2) {
    List<OnlineLearnerParams> params = new ArrayList<OnlineLearnerParams>();
    for (EtaUpdate etaUpdate : etaUpdates) {
      for (float lambda : lambdas) {
        OnlineLearnerParams.Builder paramsBuilder = OnlineLearnerParams.builder()
            .etaUpdate(etaUpdate);
        if (l2) {
          paramsBuilder.L2(lambda);
        } else {
          paramsBuilder.L1(lambda, 20);
        }
        params.add(paramsBuilder.build());
      }
    }
    return params;
  }
  
  @Override
  public String getDescription() {
    return "Fits a set of classification models to a labeled dataset";
  }
}
