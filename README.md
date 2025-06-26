```shell
devbox shell
npm install
npm run dev
```

```javascript
 // Get data
const ret = await db.query(`
  SELECT * from todo WHERE id = 1;
`)
console.log(ret.rows)

// Ask model
model
    .invoke([ new HumanMessage({ content: "What is 1 + 1?" }) ])
    .then(response => {
        console.log("Ответ модели:", response);
    })
    .catch(error => {
        console.error("Ошибка при вызове модели:", error);
    });

// Add model with RAG

let inputs = { question: "What tasks are available in the repository?" };

const result = await graph.invoke(inputs);
console.log(result.answer);

```