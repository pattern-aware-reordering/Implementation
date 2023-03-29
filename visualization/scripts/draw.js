function drawMatrixPixi(
    information,
    cells,
    parent,
    N,
    configs,
    highlightCallback = () => {},
    clearCallback = () => {}
) {
    let {
        width,
        height,
        cellWidth,
        cellHeight,
        left,
        right,
        top,
        bottom,
        lineWidth,
    } = configs

    if (lineWidth == undefined) lineWidth = 1
    if (cellWidth <= 2 || cellHeight <= 2) lineWidth = 0

    const app = new PIXI.Application({
        width,
        height,
        backgroundColor: 0xffffff,
    })

    parent.appendChild(app.view)
    const graphics = new PIXI.Graphics()

    const matrix = new Array(N).fill(0).map((_) => new Array(N).fill(0))

    // 用以处理hover时候的遮盖
    let hoverMask = new PIXI.Graphics()
    const showMask = (start, end) => {
        // 遮盖未选中的区域
        hoverMask.beginFill(0xffffff, 0.8)
        hoverMask.drawRect(left, top, cellWidth * start, cellHeight * start)
        hoverMask.drawRect(
            left,
            top + cellHeight * end,
            cellWidth * start,
            cellHeight * (N - end)
        )
        hoverMask.drawRect(
            left + cellWidth * end,
            top,
            cellWidth * (N - end),
            cellHeight * start
        )
        hoverMask.drawRect(
            left + cellWidth * end,
            top + cellHeight * end,
            cellWidth * (N - end),
            cellHeight * (N - end)
        )
        hoverMask.endFill()

        // 高亮边缘线
        hoverMask.lineStyle(lineWidth, 0x000000, 1)
        hoverMask.moveTo(left + cellWidth * start, top)
        hoverMask.lineTo(left + cellWidth * start, bottom)
        hoverMask.moveTo(left + cellWidth * end, top)
        hoverMask.lineTo(left + cellWidth * end, bottom)
        hoverMask.moveTo(left, top + cellHeight * start)
        hoverMask.lineTo(right, top + cellHeight * start)
        hoverMask.moveTo(left, top + cellHeight * end)
        hoverMask.lineTo(right, top + cellHeight * end)

        matrix.forEach((row, i) => {
            if (start <= i && i < end) {
                row.forEach((rect, j) => {
                    if (start <= j && j < end) {
                        if (rect) rect.highlight()
                    }
                })
            }
        })
        highlightCallback(start, end)
    }

    const clearMask = (start = 0, end = N) => {
        // 清空遮罩
        hoverMask.clear()
        matrix.forEach((row, i) => {
            if (start <= i && i < end) {
                row.forEach((rect, j) => {
                    if (start <= j && j < end) {
                        if (rect) rect.unhighlight()
                    }
                })
            }
        })

        clearCallback(start, end)
    }

    // 不考虑节点交互，用同一个graphics画，用以提升效率
    // graphics.beginFill(0x000000);
    // cells.forEach(([i, j]) => {
    //     graphics.drawRect(left + i * cellWidth, top + j * cellHeight, cellWidth, cellHeight);
    // })
    // graphics.endFill();

    // 考虑节点交互，设置单独的graphics
    const container = new PIXI.Container()
    cells.forEach(([i, j]) => {
        const rect = new PIXI.Graphics()
            .beginFill(0x000000)
            .drawRect(
                left + i * cellWidth,
                top + j * cellHeight,
                cellWidth,
                cellHeight
            )
            .endFill()

        rect.interactive = true
        // rect.on("pointerover", function () {
        //     // TODO: 节点的回调函数
        //     showMask(i, i + 1, j, j + 1)
        // })
        // rect.on("pointerout", function () {
        //     // TODO: 节点的回调函数
        //     clearMask(i, i + 1, j, j + 1)
        // })
        matrix[i][j] = rect
        rect.highlight = () => {
            rect.clear()
                .beginFill(0xff0000)
                .drawRect(
                    left + i * cellWidth,
                    top + j * cellHeight,
                    cellWidth,
                    cellHeight
                )
                .endFill()
        }
        rect.unhighlight = () => {
            rect.clear()
                .beginFill(0x000000)
                .drawRect(
                    left + i * cellWidth,
                    top + j * cellHeight,
                    cellWidth,
                    cellHeight
                )
                .endFill()
        }
        container.addChild(rect)
    })
    app.stage.addChild(container)

    // 画网格线
    graphics.lineStyle(lineWidth, 0xe0e0e0, 1, 0.5, true)

    for (let i = 0; i <= N; i++) {
        graphics.moveTo(left + i * cellWidth, top)
        graphics.lineTo(left + i * cellWidth, bottom)
    }
    for (let j = 0; j <= N; j++) {
        graphics.moveTo(left, top + j * cellHeight)
        graphics.lineTo(right, top + j * cellHeight)
    }

    app.stage.addChild(graphics)

    // 监听个体有可能有效率损失，如果太慢，索性全局监听，自己根据位置算
    // app.stage.interactive = true;
    // app.stage.hitArea = app.renderer.screen;

    // app.stage.addListener('mousemove', (e) => {
    // console.log(e)
    // });

    // cluster bar
    if ("clustering" in information) {
        const node2cluster = {} // 节点id映射到cluster的index上
        information.clustering.forEach((cluster, i) => {
            cluster.forEach((node) => {
                node2cluster[node] = i
            })
        })

        let clusterBars = [] // 用于在边上画一些bar
        for (let i = 1; i <= N; i++) {
            if (
                node2cluster[information.nodes[i]] !==
                    node2cluster[information.nodes[i - 1]] ||
                i == N
            ) {
                const lastBarWidth =
                    clusterBars[clusterBars.length - 2]?.width ?? 0
                const lastBarX = clusterBars[clusterBars.length - 2]?.x ?? left
                const barWidth = left + i * cellWidth - lastBarWidth - lastBarX
                const barX = lastBarX + lastBarWidth
                const barHeight = cellHeight * 2
                clusterBars.push({
                    // 横向的bar
                    start: clusterBars[clusterBars.length - 2]?.end ?? 0,
                    end: i,
                    width: barWidth,
                    height: barHeight,
                    x: barX,
                    y: top - barHeight * 1.5,
                    cluster: node2cluster[information.nodes[i - 1]],
                })
                clusterBars.push({
                    // 纵向的bar
                    start: clusterBars[clusterBars.length - 2]?.end ?? 0,
                    end: i,
                    width: barHeight,
                    height: barWidth,
                    x: top - barHeight * 1.5,
                    y: barX,
                    cluster: node2cluster[information.nodes[i - 1]],
                })
            }
        }
        information.clustering = information.clustering.map(
            (cluster) => new Set(cluster)
        )
        clusterBars = clusterBars.filter(
            (clusterBar) => information.clustering[clusterBar.cluster].size > 1
        )

        const onMouseOverBar = (bar) => {
            showMask(bar.start, bar.end)
        }
        const onMouseOutBar = (bar) => {
            clearMask()
        }

        clusterBars.forEach((bar) => {
            const rect = new PIXI.Graphics()
                .lineStyle(2, 0x9f9f9f)
                .beginFill(0xf2f2f2)
                .drawRect(bar.x, bar.y, bar.width, bar.height)

            rect.interactive = true
            rect.on("pointerover", () => {
                onMouseOverBar(bar)
            })
            rect.on("pointerout", () => {
                onMouseOutBar(bar)
            })
            app.stage.addChild(rect)
        })
    }

    app.stage.addChild(hoverMask)

    // /************** ********* **************/
    // /************** add brush **************/
    // /************** ********* **************/
    // const extent = [
    //     [width * size, 0],
    //     [right, height * size],
    // ]
    // const brush = d3
    //     .brushY()
    //     .extent(extent)
    //     .on("start", brushstart)
    //     .on("brush", brushed)
    //     .on("end", brushend)

    // const svg = d3
    //     .select(parent)
    //     .append("svg")
    //     .style("position", "absolute")
    //     .attr("width", `${width - left}px`)
    //     .attr("height", `${height - top}px`)
    //     .style("left", `${left}px`)
    //     .style("top", `${top}px`)

    // svg.append("rect")
    //     .attr("width", extent[1][0] - extent[0][0])
    //     .attr("height", extent[1][1] - extent[0][1])
    //     .attr("x", extent[0][0])
    //     .attr("y", extent[0][1])
    //     .attr("fill", "#ddd")
    // svg.selectAll("line.row")
    //     .data(d3.range(N - 1))
    //     .enter()
    //     .append("line")
    //     .classed("row", true)
    //     .attr("x1", extent[0][0])
    //     .attr("x2", extent[1][0])
    //     .attr("y1", (i) => extent[0][1] + i * cellHeight)
    //     .attr("y2", (i) => extent[0][1] + i * cellHeight)
    //     .attr("stroke", "transparent")

    // svg.append("g").call(brush)

    // const selectedRanges = [0, 0]
    // function brushstart(event) {
    //     const selection = event.selection
    //     if (!event.sourceEvent || !selection) return
    //     clearCallback()
    //     matrix.forEach((row) =>
    //         row.forEach((cell) => {
    //             if (cell) cell.unhighlight()
    //         })
    //     )
    // }

    // function brushed(event) {
    //     const selection = event.selection
    //     if (!event.sourceEvent || !selection) return
    //     console.log(selection)
    //     let [y0, y1] = selection
    //     const [start, end] = [
    //         Math.floor(y0 / cellHeight),
    //         Math.ceil(y1 / cellHeight),
    //     ]
    //     y0 = start * cellHeight
    //     y1 = end * cellHeight
    //     // Math.max(x1 - x0, y1 - y0)
    //     d3.select(this).call(brush.move, [y0, y1])
    //     matrix.forEach((row, i) =>
    //         row.forEach((cell, j) => {
    //             const isSelected =
    //                 i >= start && i < end && j >= start && j < end
    //             if (cell) {
    //                 if (isSelected) {
    //                     cell.highlight()
    //                 } else {
    //                     cell.unhighlight()
    //                 }
    //             }
    //         })
    //     )
    // }
    // function brushend(event) {
    //     const selection = event.selection
    //     if (!event.sourceEvent || !selection) return
    //     let [y0, y1] = selection
    //     const [start, end] = [
    //         Math.floor(y0 / cellHeight),
    //         Math.ceil(y1 / cellHeight),
    //     ]
    //     highlightCallback(start, end)
    // }
    // /************** ********* **************/
    // /************** add brush **************/
    // /************** ********* **************/

    matrix.canvas = app.view
    return matrix
}
