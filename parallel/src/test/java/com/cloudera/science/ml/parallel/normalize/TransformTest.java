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
package com.cloudera.science.ml.parallel.normalize;

import com.cloudera.science.ml.core.summary.Numeric;
import com.cloudera.science.ml.core.summary.SummaryStats;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class TransformTest {
  @Test
  public void testBasic() throws Exception {
    SummaryStats ss = new SummaryStats("foo", new Numeric(0.0, 1.0, 0.5, 0.1));
    assertEquals(0.0, Transform.NONE.apply(-1.0, ss), 0.001);
    assertEquals(0.2, Transform.NONE.apply(0.2, ss), 0.001);
    assertEquals(1.0, Transform.NONE.apply(10.3, ss), 0.001);
  }

  @Test
  public void testLog() throws Exception {
    SummaryStats ss = new SummaryStats("foo", new Numeric(0.0, 1.0, 0.5, 0.1));
    assertEquals(0.0, Transform.LOG.apply(-1.0, ss), 0.001);
    assertEquals(0.0, Transform.LOG.apply(0.0, ss), 0.001);
    assertEquals(Math.log(2.0), Transform.LOG.apply(1.0, ss), 0.001);
  }
}
