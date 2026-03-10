"use client"

import { useMemo, useState } from "react"
import Controller from "@/components/controller"
import {
  clampToGridCoordinate,
  getGridCoordinateRange,
} from "@/components/math"
import MinkowskiViewer from "@/components/minkowski-viewer"
import type { EtaSignature } from "@/components/types"

const GRID_PRESETS = [9, 17, 25, 33, 41]
const DIM_PRESETS = [96, 192, 384, 768]
const THETA_PRESETS = [2, 4, 8, 16, 32, 64]
const PHI_PRESETS = [2, 4, 8, 16, 32, 64]

function cyclePreset(
  values: readonly number[],
  current: number,
  direction: -1 | 1
) {
  const currentIndex = values.indexOf(current)
  const safeIndex = currentIndex >= 0 ? currentIndex : 0
  const nextIndex = (safeIndex + direction + values.length) % values.length
  return values[nextIndex]
}

export default function Page() {
  const [isOrtho, setIsOrtho] = useState(false)
  const [gridValue, setGridValue] = useState(25)
  const [dimValue, setDimValue] = useState(384)
  const [thetaValue, setThetaValue] = useState(16)
  const [phiValue, setPhiValue] = useState(16)
  const [extentValue, setExtentValue] = useState(27)
  const [etaValue, setEtaValue] = useState<EtaSignature>("negative-positive")
  const [tValue, setTValue] = useState(0)
  const [xValue, setXValue] = useState(0)
  const [yValue, setYValue] = useState(0)

  const queryRange = useMemo(
    () => getGridCoordinateRange(gridValue),
    [gridValue]
  )

  const updateGridValue = (direction: -1 | 1) => {
    const nextGridValue = cyclePreset(GRID_PRESETS, gridValue, direction)

    setGridValue(nextGridValue)
    setXValue((current) => clampToGridCoordinate(current, nextGridValue))
    setYValue((current) => clampToGridCoordinate(current, nextGridValue))
  }

  return (
    <main className="relative h-svh w-full overflow-hidden bg-[radial-gradient(circle_at_top_left,_#172554_0%,_#020617_48%,_#01030a_100%)] text-slate-50">
      <MinkowskiViewer
        gridValue={gridValue}
        isOrtho={isOrtho}
        dimValue={dimValue}
        thetaValue={thetaValue}
        phiValue={phiValue}
        extentValue={extentValue}
        etaValue={etaValue}
        tValue={tValue}
        xValue={xValue}
        yValue={yValue}
      />

      <div className="pointer-events-none absolute inset-0">
        <div className="pointer-events-none absolute top-4 left-4 hidden max-w-sm rounded-2xl border border-white/10 bg-slate-950/35 px-4 py-3 text-sm text-slate-200 shadow-xl backdrop-blur-md md:block">
          Drag in the sketch to orbit. Scroll to zoom. The controller stays
          pinned over the canvas.
        </div>

        <div className="pointer-events-auto absolute top-4 right-4 z-10">
          <Controller
            isOrtho={isOrtho}
            onOrthoChange={setIsOrtho}
            gridValue={gridValue}
            dimValue={dimValue}
            thetaValue={thetaValue}
            phiValue={phiValue}
            extentValue={extentValue}
            etaValue={etaValue}
            tValue={tValue}
            xValue={xValue}
            yValue={yValue}
            queryMin={queryRange.min}
            queryMax={queryRange.max}
            onGridPrevious={() => updateGridValue(-1)}
            onGridNext={() => updateGridValue(1)}
            onDimPrevious={() =>
              setDimValue((current) => cyclePreset(DIM_PRESETS, current, -1))
            }
            onDimNext={() =>
              setDimValue((current) => cyclePreset(DIM_PRESETS, current, 1))
            }
            onThetaPrevious={() =>
              setThetaValue((current) =>
                cyclePreset(THETA_PRESETS, current, -1)
              )
            }
            onThetaNext={() =>
              setThetaValue((current) => cyclePreset(THETA_PRESETS, current, 1))
            }
            onPhiPrevious={() =>
              setPhiValue((current) => cyclePreset(PHI_PRESETS, current, -1))
            }
            onPhiNext={() =>
              setPhiValue((current) => cyclePreset(PHI_PRESETS, current, 1))
            }
            onExtentChange={setExtentValue}
            onEtaChange={setEtaValue}
            onTChange={setTValue}
            onXChange={setXValue}
            onYChange={setYValue}
          />
        </div>
      </div>
    </main>
  )
}
