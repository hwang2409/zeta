// Evaluated by the trusted loader outside the script's lexical scope.
(names => {
  'use strict';
  const read = Javy.IO.readSync, write = Javy.IO.writeSync;
  const encode = new TextEncoder().encode.bind(new TextEncoder());
  const decode = new TextDecoder().decode.bind(new TextDecoder());
  const stringify = JSON.stringify, parse = JSON.parse, assign = Object.assign;
  const Bytes = Uint8Array, Err = Error, P = Promise, enqueue = P.prototype.then.bind(P.resolve());
  const max = 65536;
  for (const name of ['Javy', 'console', 'Date', 'performance', 'Intl']) {
    if (!Reflect.deleteProperty(globalThis, name)) throw new Err(`cannot remove ${name}`);
  }
  delete Math.random;

  function json(value) {
    return stringify(value, (_, v) => {
      if (typeof v === 'undefined' || typeof v === 'function' || typeof v === 'symbol' ||
          (typeof v === 'number' && !Number.isFinite(v))) throw new Err('value must be JSON-compatible');
      return v;
    });
  }
  function send(frame) {
    const bytes = encode(stringify(frame) + '\n');
    if (bytes.length > max) throw new Err('request frame too large');
    if (write(1, bytes) !== bytes.length) throw new Err('short bridge write');
  }
  function receive() {
    const buffer = new Bytes(max);
    function fill(length) {
      for (let i = 0; i < length;) {
        const n = read(0, buffer.subarray(i, length));
        if (n <= 0 || n > length - i) throw new Err('short bridge read');
        i += n;
      }
    }
    fill(4);
    const length = buffer[0] + 256 * buffer[1] + 65536 * buffer[2] + 16777216 * buffer[3];
    if (length > max) throw new Err('response frame too large');
    fill(length);
    return parse(decode(buffer.subarray(0, length)));
  }

  let queue = [], emitted = false;
  function flush() {
    const batch = queue;
    queue = [];
    try {
      if (batch.length !== 1) throw new Err('code mode only accepts sequential tool calls');
      const item = batch[0];
      send({op: 'call', calls: [item.request]});
      const [reply] = receive();
      if (reply.ok) item.resolve(reply.value);
      else item.reject(assign(new Err(reply.error.content), reply.error));
    } catch (error) {
      for (const item of batch) item.reject(error);
    }
  }
  function call(tool, args) {
    return new P((resolve, reject) => {
      const request = parse(json({tool, args}));
      queue.push({request, resolve, reject});
      if (queue.length === 1) enqueue(flush);
    });
  }
  const tools = {call};
  for (const name of names) tools[name] = args => call(name, args);
  Object.defineProperties(globalThis, {
    tools: {value: Object.freeze(tools)},
    output: {value: value => {
      if (emitted) throw new Err('output can only be called once');
      const text = json(value);
      if (encode(text).length > 1048576) throw new Err('output exceeds 1 MiB');
      emitted = true;
      // Split the serialized value to keep each escaped transport frame bounded.
      for (let i = 0; i < text.length; i += 8192) {
        send({op: 'output', text: text.slice(i, i + 8192), end: i + 8192 >= text.length});
      }
    }},
  });
})
