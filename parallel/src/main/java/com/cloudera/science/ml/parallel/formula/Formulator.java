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
package com.cloudera.science.ml.parallel.formula;

import org.apache.crunch.MapFn;
import org.apache.crunch.PCollection;
import org.apache.mahout.math.Vector;

import com.cloudera.science.ml.core.formula.Formula;
import com.cloudera.science.ml.core.records.Record;
import com.cloudera.science.ml.parallel.types.MLAvros;
import com.google.common.base.Function;

/**
 * Use a {@code Formula} instance to convert a {@code PCollection<Record>} into a
 * {@code PCollection<Vector>} instance.
 */
public class Formulator implements Function<PCollection<Record>, PCollection<Vector>> {
  
  private final Formula formula;
  
  public Formulator(Formula formula) {
    this.formula = formula;
  }
  
  @Override
  public PCollection<Vector> apply(PCollection<Record> records) {
    return records.parallelDo(new FormulaFn(formula), MLAvros.vector());
  }
  
  private static class FormulaFn extends MapFn<Record, Vector> {

    private final Formula formula;
    
    public FormulaFn(Formula formula) {
      this.formula = formula;
    }
    
    @Override
    public Vector map(Record r) {
      return formula.apply(r);
    }
    
  }
}
