const realUtil = require('node:util');

function isNullOrUndefined(value) { return value === null || value === undefined; }

module.exports = Object.assign({}, realUtil, { isNullOrUndefined });

