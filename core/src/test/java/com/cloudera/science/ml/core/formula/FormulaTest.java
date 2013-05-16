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
package com.cloudera.science.ml.core.formula;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.cloudera.science.ml.core.records.DataType;
import com.cloudera.science.ml.core.records.Record;
import com.cloudera.science.ml.core.records.RecordSpec;
import com.cloudera.science.ml.core.records.SimpleRecord;
import com.cloudera.science.ml.core.records.Spec;
import com.cloudera.science.ml.core.summary.Numeric;
import com.cloudera.science.ml.core.summary.Summary;
import com.cloudera.science.ml.core.summary.SummaryBuilder;
import com.cloudera.science.ml.core.vectors.Vectors;
import com.google.common.collect.ImmutableList;

/**
 *
 */
public class FormulaTest {

  private Spec abSpec = RecordSpec.builder()
      .add("a", DataType.DOUBLE)
      .add("b", DataType.STRING)
      .build();
  
  @Test
  public void testBasic() {
    Summary summary = new SummaryBuilder(abSpec)
        .categorical("b", ImmutableList.of("x", "y", "z"))
        .numeric("a", new Numeric(17.29, 17.29, 17.29, 0.0))
        .build();
    Formula f = Formula.compile(ImmutableList.of(Term.INTERCEPT, new Term("a"), new Term("a", "b")),
        summary);
    
    Record r = new SimpleRecord(abSpec, 17.29, "z");
    assertEquals(Vectors.of(1.0, 17.29, 0.0, 17.29), f.apply(r));
    r = new SimpleRecord(abSpec, 1.2, "x");
    assertEquals(Vectors.of(1.0, 1.2, 0.0, 0.0), f.apply(r));
    r = new SimpleRecord(abSpec, 2, "y");
    assertEquals(Vectors.of(1.0, 2.0, 2.0, 0.0), f.apply(r));
  }
}
