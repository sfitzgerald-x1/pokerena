#!/usr/bin/env node

const {calculate, Field, Generations, Move, Pokemon} = require('@smogon/calc');

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

function readStdin() {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', chunk => {
      data += chunk;
    });
    process.stdin.on('end', () => resolve(data));
    process.stdin.on('error', reject);
  });
}

function requireObject(value, label) {
  if (!value || Array.isArray(value) || typeof value !== 'object') {
    fail(`${label} must be a JSON object.`);
  }
  return value;
}

function requireString(value, label) {
  if (typeof value !== 'string' || !value.trim()) {
    fail(`${label} must be a non-empty string.`);
  }
  return value;
}

function optionalObject(value, label) {
  if (value === undefined || value === null) {
    return {};
  }
  return requireObject(value, label);
}

function requireSchemaVersion(value) {
  if (value !== 'pokerena.damage-request.v1') {
    fail("schema_version must be 'pokerena.damage-request.v1'.");
  }
}

function normalizeDamage(damage) {
  if (typeof damage === 'number') {
    return damage;
  }
  if (Array.isArray(damage)) {
    return damage.map(normalizeDamage);
  }
  fail('Damage result was not a number or array.');
}

function percent(value, total) {
  if (!total) {
    return 0;
  }
  return Number(((value / total) * 100).toFixed(2));
}

function buildPokemon(gen, value, label) {
  const data = requireObject(value, label);
  const species = requireString(data.species, `${label}.species`);
  const options = optionalObject(data.options, `${label}.options`);
  return new Pokemon(gen, species, options);
}

function buildMove(gen, value) {
  const data = requireObject(value, 'move');
  const name = requireString(data.name, 'move.name');
  const options = optionalObject(data.options, 'move.options');
  return new Move(gen, name, options);
}

async function main() {
  const rawInput = (await readStdin()).trim();
  if (!rawInput) {
    fail('Expected a JSON payload on stdin.');
  }

  let payload;
  try {
    payload = JSON.parse(rawInput);
  } catch (error) {
    fail(`Failed to parse JSON input: ${error.message}`);
  }

  const data = requireObject(payload, 'payload');
  requireSchemaVersion(data.schema_version);
  const generation = data.generation;
  if (!Number.isInteger(generation) || generation < 1) {
    fail('generation must be a positive integer.');
  }

  const gen = Generations.get(generation);
  if (!gen) {
    fail(`Unsupported generation: ${generation}`);
  }

  const attacker = buildPokemon(gen, data.attacker, 'attacker');
  const defender = buildPokemon(gen, data.defender, 'defender');
  const move = buildMove(gen, data.move);
  const field = new Field(optionalObject(data.field, 'field'));
  const result = calculate(gen, attacker, defender, move, field);
  const range = result.range();
  const defenderHP = defender.maxHP();
  const kochance = result.kochance();

  const response = {
    schema_version: 'pokerena.damage-result.v1',
    generation: gen.num,
    attacker: {
      species: attacker.name,
      level: attacker.level,
    },
    defender: {
      species: defender.name,
      level: defender.level,
      hp: defenderHP,
    },
    move: {
      name: move.name,
    },
    damage: normalizeDamage(result.damage),
    range: {
      min: range[0],
      max: range[1],
    },
    range_percent: {
      min: percent(range[0], defenderHP),
      max: percent(range[1], defenderHP),
    },
    description: result.desc(),
    knockout: {
      chance: kochance.chance ?? null,
      hits: kochance.n,
      text: kochance.text,
    },
  };

  process.stdout.write(`${JSON.stringify(response)}\n`);
}

main().catch(error => {
  fail(error instanceof Error ? error.message : String(error));
});
