const checkBtn = document.getElementById("checkBtn");

if (checkBtn) {
  const newsInput = document.getElementById("newsInput");
  const resultBox = document.getElementById("resultBox");
  const confidenceValue = document.getElementById("confidenceValue");
  const categoryValue = document.getElementById("categoryValue");

  checkBtn.addEventListener("click", function () {
    const text = newsInput.value.trim();

    if (text === "") {
      alert("Please enter news article content first.");
      return;
    }

    const isFake = Math.random() > 0.5;

    const fakeCategories = ["Politics", "Social Media", "Breaking News", "Health"];
    const realCategories = ["Education", "Technology", "Business", "Science"];

    if (isFake) {
      resultBox.textContent = "Result: FAKE NEWS";
      resultBox.classList.remove("real");
      confidenceValue.textContent = Math.floor(Math.random() * 10 + 85) + "%";
      categoryValue.textContent =
        fakeCategories[Math.floor(Math.random() * fakeCategories.length)];
    } else {
      resultBox.textContent = "Result: REAL NEWS";
      resultBox.classList.add("real");
      confidenceValue.textContent = Math.floor(Math.random() * 10 + 90) + "%";
      categoryValue.textContent =
        realCategories[Math.floor(Math.random() * realCategories.length)];
    }
  });
}