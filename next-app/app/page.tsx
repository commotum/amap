"use client"

import { PanelRightOpen } from "lucide-react"
import { useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import Controller from "@/components/controller"
import {
  clampToGridCoordinate,
  getGridCoordinateRange,
  getGridCoordinateStep,
} from "@/components/math"
import MinkowskiViewer from "@/components/minkowski-viewer"
import type { EtaSignature } from "@/components/types"

const GRID_PRESETS = [4, 8, 16, 32, 64, 128, 256, 512]
const DIM_PRESETS = [96, 192, 384, 768]
const THETA_PRESETS = [1000, 5000, 10000, 20000, 30000, 45000, 60000, 80000, 100000]
const PHI_PRESETS = [1000, 5000, 10000, 20000, 30000, 45000, 60000, 80000, 100000]
const INITIAL_GRID_VALUE = 32
const DEFAULT_QUERY_BY_GRID: Record<number, { x: number; y: number }> = {
  4: { x: -0.5, y: 0.5 },
  8: { x: -1, y: 1 },
  16: { x: -2, y: 2 },
  32: { x: -4, y: 4 },
  64: { x: -7.5, y: 7.5 },
  128: { x: -15, y: 15 },
  256: { x: -30.5, y: 30.5 },
  512: { x: -60.5, y: 60.5 },
}

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

function getDefaultQuery(gridValue: number) {
  const presetQuery = DEFAULT_QUERY_BY_GRID[gridValue]

  if (presetQuery) {
    return {
      x: clampToGridCoordinate(presetQuery.x, gridValue),
      y: clampToGridCoordinate(presetQuery.y, gridValue),
    }
  }

  return {
    x: clampToGridCoordinate(0, gridValue),
    y: clampToGridCoordinate(0, gridValue),
  }
}

export default function Page() {
  const [isOrtho, setIsOrtho] = useState(false)
  const [isControllerCollapsed, setIsControllerCollapsed] = useState(false)
  const [isControllerHovered, setIsControllerHovered] = useState(false)
  const [isControllerFocused, setIsControllerFocused] = useState(false)
  const [resetViewKey, setResetViewKey] = useState(0)
  const [gridValue, setGridValue] = useState(INITIAL_GRID_VALUE)
  const [dimValue, setDimValue] = useState(384)
  const [thetaValue, setThetaValue] = useState(10000)
  const [phiValue, setPhiValue] = useState(10000)
  const [extentValue, setExtentValue] = useState(6.28)
  const [etaValue, setEtaValue] = useState<EtaSignature>("negative-positive")
  const [tValue, setTValue] = useState(0)
  const [xValue, setXValue] = useState(() => getDefaultQuery(INITIAL_GRID_VALUE).x)
  const [yValue, setYValue] = useState(() => getDefaultQuery(INITIAL_GRID_VALUE).y)

  const queryRange = useMemo(
    () => getGridCoordinateRange(gridValue),
    [gridValue]
  )
  const queryStep = getGridCoordinateStep()
  const isControllerActive = isControllerHovered || isControllerFocused

  const updateGridValue = (direction: -1 | 1) => {
    const nextGridValue = cyclePreset(GRID_PRESETS, gridValue, direction)
    const nextQuery = getDefaultQuery(nextGridValue)

    setGridValue(nextGridValue)
    setXValue(nextQuery.x)
    setYValue(nextQuery.y)
  }

  return (
    <main className="relative h-svh w-full overflow-hidden bg-[radial-gradient(circle_at_top_left,_#172554_0%,_#020617_48%,_#01030a_100%)] text-slate-50">
      <MinkowskiViewer
        gridValue={gridValue}
        isOrtho={isOrtho}
        orbitEnabled={!isControllerActive}
        resetViewKey={resetViewKey}
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

        <div
          className="pointer-events-auto absolute top-4 right-4 z-10"
          onPointerEnter={() => setIsControllerHovered(true)}
          onPointerLeave={() => setIsControllerHovered(false)}
          onFocusCapture={() => setIsControllerFocused(true)}
          onBlurCapture={(event) => {
            const nextTarget = event.relatedTarget

            if (!(nextTarget instanceof Node) || !event.currentTarget.contains(nextTarget)) {
              setIsControllerFocused(false)
            }
          }}
        >
          {isControllerCollapsed ? (
            <Button
              type="button"
              variant="outline"
              size="icon"
              className="border-white/10 bg-slate-950/72 text-slate-50 shadow-2xl backdrop-blur-xl hover:bg-slate-900"
              onClick={() => setIsControllerCollapsed(false)}
              aria-label="Expand controller"
              title="Expand controller"
            >
              <PanelRightOpen />
            </Button>
          ) : (
            <Controller
              isOrtho={isOrtho}
              onResetView={() => setResetViewKey((current) => current + 1)}
              onCollapse={() => setIsControllerCollapsed(true)}
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
              queryStep={queryStep}
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
                setThetaValue((current) =>
                  cyclePreset(THETA_PRESETS, current, 1)
                )
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
              onXChange={(value) =>
                setXValue(clampToGridCoordinate(value, gridValue))
              }
              onYChange={(value) =>
                setYValue(clampToGridCoordinate(value, gridValue))
              }
            />
          )}
        </div>
      </div>
    </main>
  )
}
