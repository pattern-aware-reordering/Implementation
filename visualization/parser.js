async function parseMatrix(filePath) {
    const response = await fetch(filePath)
    const promise = response.json()
    const content = await promise.then()
    const matrix = content.matrix
    const information = content.information
    return { information, matrix }
}
