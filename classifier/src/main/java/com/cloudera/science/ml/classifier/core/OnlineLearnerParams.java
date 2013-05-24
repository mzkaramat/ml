package com.cloudera.science.ml.classifier.core;

import java.io.Serializable;

import com.google.common.base.Preconditions;

public class OnlineLearnerParams implements Serializable {

  private final double lambda;
  private final EtaUpdate etaUpdate;
  private final boolean pegasos;
  private final double l1Radius;
  private final int l1Iterations;

  private OnlineLearnerParams(double lambda, EtaUpdate etaUpdate,
      boolean pegasos, double l1Radius, int l1Iterations) {
    this.lambda = lambda;
    this.etaUpdate = etaUpdate;
    this.pegasos = pegasos;
    this.l1Radius = l1Radius;
    this.l1Iterations = l1Iterations;
  }

  public static OnlineLearnerParams.Builder builder() {
    return new Builder();
  }

  public double lambda() {
    return lambda;
  }

  public double eta(int iteration) {
    return etaUpdate.compute(lambda, iteration);
  }

  public EtaUpdate etaUpdate() {
    return etaUpdate();
  }
  
  public double l1Radius() {
    return l1Radius;
  }
  
  public int l1Iterations() {
    return l1Iterations;
  }
  
  public boolean pegasos() {
    return pegasos;
  }

  public void updateWeights(WeightVector weights, int iteration) {
    if (pegasos) {
      weights.pegasosProjection(lambda);
    }
    if (l1Iterations > 0 && iteration % l1Iterations == 0) {
      weights.approxRegularizeL1(l1Radius);
    }
  }
  
  @Override
  public String toString() {
    return "[lambda=" + lambda + ", etaUpdate=" + etaUpdate.getClass().getSimpleName()
        + ", pegasos=" + pegasos + ", l1Radius=" + l1Radius + ", l1Iterations="
        + l1Iterations + "]";
  }

  public static class Builder {
    private double lambda = 0.0;
    private EtaUpdate etaUpdate = EtaUpdate.BASIC_ETA;
    private boolean pegasos = false;
    private int l1Iterations = -1;
    private double l1Radius = 10.0;

    public OnlineLearnerParams.Builder L2(double lambda) {
      Preconditions.checkArgument(lambda >= 0.0);
      this.lambda = lambda;
      return this;
    }

    public OnlineLearnerParams.Builder etaUpdate(EtaUpdate etaUpdate) {
      this.etaUpdate = Preconditions.checkNotNull(etaUpdate);
      return this;
    }

    public OnlineLearnerParams.Builder pegasos(boolean pegasos) {
      this.pegasos = pegasos;
      return this;
    }

    public OnlineLearnerParams.Builder L1(double radius, int iterations) {
      Preconditions.checkArgument(iterations > 0, "L1 iterations must be > 0");
      Preconditions.checkArgument(radius > 0.0, "L1 radius must be > 0");
      this.l1Radius = radius;
      this.l1Iterations = iterations;
      return this;
    }

    public OnlineLearnerParams build() {
      // Do some checking for problematic interactions
      if (lambda == 0.0) {
        if (etaUpdate == EtaUpdate.PEGASOS_ETA) {
          throw new IllegalStateException("PEGASOS_ETA requires L2 lambda > 0");
        }
        if (pegasos) {
          throw new IllegalStateException("Pegasos projection requires L2 lambda > 0");
        }
      }
      return new OnlineLearnerParams(lambda, etaUpdate, pegasos, l1Radius, l1Iterations);
    }
  }
}