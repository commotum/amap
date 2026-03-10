export type EtaSignature = "negative-positive" | "positive-negative"

export interface MinkowskiControls {
  gridValue: number
  dimValue: number
  thetaValue: number
  phiValue: number
  extentValue: number
  etaValue: EtaSignature
  tValue: number
  xValue: number
  yValue: number
}
