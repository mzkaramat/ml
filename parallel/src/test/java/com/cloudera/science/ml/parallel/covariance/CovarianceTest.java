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
package com.cloudera.science.ml.parallel.covariance;

import com.cloudera.science.ml.core.vectors.Vectors;
import com.cloudera.science.ml.parallel.types.MLAvros;
import com.google.common.collect.Lists;
import org.apache.crunch.PCollection;
import org.apache.crunch.PTable;
import org.apache.crunch.Pair;
import org.apache.crunch.impl.mem.MemPipeline;
import org.apache.crunch.types.avro.Avros;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class CovarianceTest {

  PCollection<Vector> VECTORS = MemPipeline.collectionOf(
      Vectors.of(1.0, 2.0, 3.0),
      Vectors.of(0.0, 1.0, 2.0),
      Vectors.of(-1, 0, 4),
      Vectors.of(17.0, 29.0, 1729.0));

  PTable<Integer, Vector> TAGGED_VECTORS = MemPipeline.typedTableOf(
      Avros.tableOf(Avros.ints(), MLAvros.vector()),
      1, Vectors.of(1.0, 2.0, 3.0),
      1, Vectors.of(0.0, 1.0, 2.0),
      1, Vectors.of(-1, 0, 4),
      1, Vectors.of(17.0, 29.0, 1729.0),
      2, Vectors.of(1.0, 2.0, 3.0));

  @Test
  public void testPCollection() throws Exception {
    PTable<Index, CoMoment> pt = Covariance.cov(VECTORS);
    assertEquals(6, Lists.newArrayList(pt.materialize()).size());
  }

  @Test
  public void testPTable() throws Exception {
    PTable<Pair<Integer, Index>, CoMoment> pt = Covariance.cov(TAGGED_VECTORS);
    assertEquals(12, Lists.newArrayList(pt.materialize()).size());
  }
}
