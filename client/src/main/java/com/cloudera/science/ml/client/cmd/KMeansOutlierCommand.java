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

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.beust.jcommander.ParametersDelegate;
import com.beust.jcommander.converters.CommaParameterSplitter;
import com.beust.jcommander.converters.IntegerConverter;
import com.beust.jcommander.internal.Maps;
import com.cloudera.science.ml.avro.MLClusterCovariance;
import com.cloudera.science.ml.client.params.CentersParameters;
import com.cloudera.science.ml.client.params.PipelineParameters;
import com.cloudera.science.ml.client.params.RecordOutputParameters;
import com.cloudera.science.ml.client.params.VectorInputParameters;
import com.cloudera.science.ml.client.util.AvroIO;
import com.cloudera.science.ml.core.matrix.Inverter;
import com.cloudera.science.ml.core.matrix.MatrixUtils;
import com.cloudera.science.ml.core.vectors.Centers;
import com.cloudera.science.ml.kmeans.parallel.ClusterKey;
import com.cloudera.science.ml.kmeans.parallel.KMeansParallel;
import com.cloudera.science.ml.parallel.covariance.MahalanobisDistance;
import com.cloudera.science.ml.parallel.records.Records;
import com.google.common.collect.ImmutableMultiset;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.crunch.PCollection;
import org.apache.crunch.Pipeline;
import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.math.NamedVector;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

@Parameters(
    commandDescription = "Uses the Mahalanobis distance with the output of kcovar to find outliers")
public class KMeansOutlierCommand implements Command {

  @Parameter(names = "--covariance-file", required=true,
      description = "The local file with the Avro-formatted covariance data from kcovar")
  private String covFile;

  @Parameter(names = "--approx",
      description = "Enable approximate center assignments to speed up processing at the cost of accuracy")
  private boolean approx = false;

  @Parameter(names = "--output-path", required=true,
      description = "The path to write the output to (id, clustering_id, center_id, outlier_distance, pvalue)")
  private String outliersPath;

  @ParametersDelegate
  private VectorInputParameters inputParams = new VectorInputParameters();

  @ParametersDelegate
  private PipelineParameters pipelineParams = new PipelineParameters();

  @ParametersDelegate
  private CentersParameters centersParams = new CentersParameters();

  @ParametersDelegate
  private RecordOutputParameters outputParams = new RecordOutputParameters();

  @Override
  public int execute(Configuration conf) throws IOException {

    List<MLClusterCovariance> ccov = AvroIO.read(MLClusterCovariance.class, new File(covFile));
    Map<ClusterKey, MahalanobisDistance> distances = Maps.newHashMap();
    for (MLClusterCovariance cc : ccov) {
      int dim = cc.getMeans().size();
      RealMatrix m = MatrixUtils.toRealMatrix(dim, dim, cc.getCov(), true);
      RealMatrix im = Inverter.SVD.apply(m); // pseudo-inverse works fine for this
      MahalanobisDistance md = new MahalanobisDistance(toArray(cc.getMeans()), im.getData(), cc.getCount());
      distances.put(new ClusterKey(cc.getClusteringId(), cc.getCenterId()), md);
    }

    Pipeline p = pipelineParams.create(KMeansOutlierCommand.class, conf);
    PCollection<NamedVector> vecs = inputParams.getVectors(p);
    KMeansParallel kmp = new KMeansParallel();

    List<Centers> centers = centersParams.getCenters();
    List<Integer> centerIds = centersParams.getCenterIds();
    validate(centers, centerIds, distances);

    Records outliers = kmp.computeOutliers(
        vecs,
        centers,
        approx,
        centerIds,
        distances);
    outputParams.writeRecords(outliers.get(), outliers.getSpec(), outliersPath);
    p.done();
    return 0;
  }

  private void validate(
      List<Centers> centers,
      List<Integer> centerIds,
      Map<ClusterKey, MahalanobisDistance> distances) {
    ImmutableMultiset.Builder<Integer> imb = ImmutableMultiset.builder();
    for (ClusterKey ck : distances.keySet()) {
      imb.add(ck.getClusterId());
    }
    Multiset<Integer> counts = imb.build();
    for (int i = 0; i < centers.size(); i++) {
      int idx = (centerIds == null || centerIds.isEmpty()) ? i : centerIds.get(i);
      if (counts.count(idx) != centers.get(i).size()) {
        throw new IllegalArgumentException("Covariance/cluster mismatch for cluster ID: " + idx);
      }
    }
  }

  private double[] toArray(List<Double> values) {
    double[] d = new double[values.size()];
    for (int i = 0; i < d.length; i++) {
      d[i] = values.get(i);
    }
    return d;
  }

  @Override
  public String getDescription() {
    return "Uses the Mahalanobis distance with the output of kcovar to find outliers";
  }
}
