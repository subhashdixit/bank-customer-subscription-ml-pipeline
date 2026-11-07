'use strict';
const form = document.getElementById('prediction-form');
const submit = document.getElementById('submit');
const status = document.getElementById('status');
const error = document.getElementById('error');
function node(tag, text, className) {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  return element;
}
function render(data) {
  document.getElementById('result').hidden = false;
  document.getElementById('placeholder').hidden = true;
  document.getElementById('probability').textContent = (data.subscription_probability * 100).toFixed(2) + '%';
  const briefing = data.briefing;
  document.getElementById('mode').textContent = briefing.mode;
  document.getElementById('context-message').textContent = briefing.message;
  const evidence = document.getElementById('evidence');
  evidence.replaceChildren();
  const sources = new Map((briefing.sources || []).map(s => [s.id, s]));
  for (const item of briefing.product_information || []) {
    const card = node('div', '', 'evidence');
    card.append(node('p', item.text));
    const source = sources.get(item.source_id);
    if (source) card.append(node('div', source.section + ' · ' + source.version + ' · ' + source.id, 'citation'));
    evidence.append(card);
  }
  for (const [id, values] of [['followups', briefing.questions_to_confirm], ['limitations', briefing.limitations]]) {
    const list = document.getElementById(id); list.replaceChildren();
    for (const text of values || []) list.append(node('li', text));
  }
  document.getElementById('provenance').textContent = briefing.knowledge_version ? 'Knowledge: ' + briefing.knowledge_version + ' · SHA-256: ' + briefing.knowledge_sha256 : '';
  document.getElementById('raw').textContent = JSON.stringify(data, null, 2);
}
form.addEventListener('submit', async event => {
  event.preventDefault();
  submit.disabled = true; error.hidden = true;
  document.getElementById('result').hidden = true;
  status.textContent = 'Calculating the score and retrieving product evidence…';
  const values = Object.fromEntries(new FormData(form));
  const product_id = values.product_id, question = values.question;
  delete values.product_id; delete values.question;
  for (const key of ['age', 'balance', 'previous']) values[key] = Number(values[key]);
  try {
    const response = await fetch('/predict/briefing', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({customer: values, product_id, question, use_llm: false}), signal: AbortSignal.timeout(30000)});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Please check the customer fields and try again.');
    render(data); status.textContent = 'Complete. Review the estimate and supporting evidence below.';
  } catch (failure) {
    error.textContent = failure.name === 'TimeoutError' ? 'The request timed out. Please retry when the service is ready.' : failure.message;
    error.hidden = false; status.textContent = 'The request could not be completed.';
  } finally { submit.disabled = false; }
});
document.getElementById('reset').addEventListener('click', () => {
  form.reset(); error.hidden = true; document.getElementById('result').hidden = true;
  document.getElementById('placeholder').hidden = false;
  status.textContent = 'Example restored. Run the prediction when ready.';
});
