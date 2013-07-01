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
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

import org.apache.crunch.PCollection;
import org.apache.crunch.Pair;
import org.apache.crunch.Pipeline;
import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.math.Vector.Element;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.beust.jcommander.ParametersDelegate;
import com.cloudera.science.ml.client.params.PipelineParameters;
import com.cloudera.science.ml.client.params.VectorInputParameters;
import com.cloudera.science.ml.client.util.AvroIO;
import com.cloudera.science.ml.classifier.avro.MLOnlineLearnerRuns;
import com.cloudera.science.ml.classifier.core.ModelScore;
import com.cloudera.science.ml.classifier.core.OnlineLearnerParams;
import com.cloudera.science.ml.classifier.core.OnlineLearnerRun;
import com.cloudera.science.ml.classifier.core.OnlineLearnerRuns;
import com.cloudera.science.ml.classifier.core.WeightVector;
import com.cloudera.science.ml.classifier.parallel.ParallelEvaluator;
import com.cloudera.science.ml.classifier.parallel.types.ClassifierAvros;
import com.cloudera.science.ml.core.vectors.LabeledVector;
import com.google.common.base.Joiner;

@Parameters(commandDescription = "Evaluates a set of models trained by the fit command")
public class EvaluateCommand implements Command {

  @Parameter(names = "--input-file", required = true,
      description = "A local Avro file that contains models trained by the fit command")
  private String runsFile;
  
  @Parameter(names = "--output-file",
      description = "A local file to write out model scores to")
  private String outFile;
  
  @ParametersDelegate
  private PipelineParameters pipelineParams = new PipelineParameters();
  
  @ParametersDelegate
  private VectorInputParameters inputParams = new VectorInputParameters();
  
  @Override
  public int execute(Configuration conf) throws IOException {
    List<MLOnlineLearnerRuns> mlRuns = AvroIO.read(MLOnlineLearnerRuns.class, new File(runsFile));
    OnlineLearnerRuns runs = ClassifierAvros.toOnlineLearnerRuns(mlRuns.get(0));
    
    Pipeline p = pipelineParams.create(EvaluateCommand.class, conf);

    PCollection<LabeledVector> labeledVectors = inputParams.getLabeledVectors(p);
 
    ParallelEvaluator evaluator = new ParallelEvaluator();
    Collection<Pair<OnlineLearnerRun, ModelScore>> runsWithScores =
        evaluator.evaluateModels(runs, labeledVectors);
    List<Pair<OnlineLearnerRun, ModelScore>> runsWithScoresList =
        new ArrayList<Pair<OnlineLearnerRun, ModelScore>>(runsWithScores);
    Collections.sort(runsWithScoresList,
        new Comparator<Pair<OnlineLearnerRun, ModelScore>>() {
      @Override
      public int compare(Pair<OnlineLearnerRun, ModelScore> o1,
          Pair<OnlineLearnerRun, ModelScore> o2) {
        return (int)Math.signum(o2.second().getAccuracy() - o1.second().getAccuracy());
      }
    });
    
    PrintStream ps = new PrintStream(outFile);
    writeStats(runsWithScoresList, ps);
    writeStats(runsWithScoresList, System.out);
    ps.close();
    p.done();
    
    return 0;
  }
  
  private void writeStats(Collection<Pair<OnlineLearnerRun, ModelScore>> runsWithScores,
      PrintStream ps) {
    ps.println(Joiner.on(',').join("Fold", "Partition", "ParamsVersion",
        "Lambda", "EtaUpdate", "L1Radius",
        "L1Iterations", "PEGASOS", "LearnerClass",
        "NonZeroWeights", "TruePositives",
        "FalsePositives", "TrueNegatives", "FalseNegatives",
        "Precision", "Recall", "Accuracy"));
    for (Pair<OnlineLearnerRun, ModelScore> runWithScore : runsWithScores) {
      OnlineLearnerRun run = runWithScore.first();
      ModelScore score = runWithScore.second();
      OnlineLearnerParams params = run.getParams();
      ps.println(String.format("%d,%d,%d,%.4f,%s,%.4f,%d,%s,%s,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f",
          run.getFold(), run.getPartition(), run.getParamsVersion(),
          params.lambda(), params.etaUpdate(), params.l1Radius(),
          params.l1Iterations(), params.pegasos(), run.getLearnerClass().getSimpleName(),
          countNonZero(run.getClassifier().getWeights()), score.getTruePositives(),
          score.getFalsePositives(), score.getTrueNegatives(), score.getFalseNegatives(),
          score.getPrecision(), score.getRecall(), score.getAccuracy()));
    }
  }
  
  private int countNonZero(WeightVector vector) {
    int count = 0;
    Iterator<Element> elIter = vector.toVector().iterateNonZero();
    while (elIter.hasNext()) {
      if (Math.abs(elIter.next().get()) > 0.0001) {
        count++;
      }
    }
    return count;
  }
  
  @Override
  public String getDescription() {
    return "Evaluates a set of models trained by the fit command";
  }

}
