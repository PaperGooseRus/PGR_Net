import Mathlib

/-!
# PGR Neural — математическая спецификация и доказательства

СТАТУС: проект формализации; файл не был проверен Lean в среде автора,
поскольку Lean 4 и Mathlib там отсутствуют. Запустите файл в проекте с
совместимыми Lean 4 и Mathlib и проверьте все сообщения компилятора.

ЧТО ПРОВЕРЯЕТСЯ:
* размерности векторов и матриц зафиксированы в типах;
* непустое сообщение и среднее представление токенов;
* свойства ReLU и softmax над идеальными вещественными числами ℝ;
* правило отказа при оценке ниже порога;
* простая абстрактная модель разрешения изменений.

ЧТО НЕ ПРОВЕРЯЕТСЯ: обучение, правильность предсказаний, выбор argmax,
реальная арифметика float32, программа C++, WinAPI, система файлов и GUI.
Lean не проверяет содержимое проекта на компьютере и не выполняет сборку.

ЗАПУСК: скопируйте файл в проект Lean 4, в котором настроена Mathlib,
и из корня проекта выполните: lake env lean PGR_Neural_Verification.lean
Lean и Mathlib должны использовать взаимно совместимые версии. Команда
проверяет исходник, а не запускает обучение нейросети.
-/

open scoped BigOperators

namespace PGR

/-! ## 1. Векторы и матрицы

`Fin n` — индекс от 0 до n-1. Если размер равен нулю, такой индекс
создать нельзя. Вектор — функция из допустимого индекса в число.
Матрица — функция из индекса строки и столбца в число.
В отличие от массива C++, выход за границы здесь исключён типом индекса.
-/

abbrev Vec (n : Nat) := Fin n → ℝ
abbrev Mat (rows cols : Nat) := Fin rows → Fin cols → ℝ

/-- Умножение матрицы на вектор: в каждой строке суммируем произведения. -/
noncomputable def matVec {rows cols : Nat}
    (weights : Mat rows cols) (input : Vec cols) : Vec rows :=
  fun row => ∑ col : Fin cols, weights row col * input col

/-- Сложение векторов одинаковой длины. -/
noncomputable def addVec {n : Nat} (a b : Vec n) : Vec n :=
  fun i => a i + b i

/-! ## 2. Словарь и непустое сообщение

Словарь размером V хранит вектор длины D для каждого индекса токена.
`Fin (n + 1)` задаёт последовательность минимум из одного токена, поэтому
деление на её длину не является делением на ноль над ℝ.
Реальный токенизатор должен отдельно проверять входные данные и индексы.
-/

abbrev Embedding (vocabSize embeddingDim : Nat) :=
  Fin vocabSize → Vec embeddingDim

/-- Средний вектор токенов. Здесь `n + 1` — фактическая длина сообщения. -/
noncomputable def meanEmbedding {vocabSize embeddingDim n : Nat}
    (table : Embedding vocabSize embeddingDim)
    (tokens : Fin (n + 1) → Fin vocabSize) : Vec embeddingDim :=
  fun coordinate =>
    (∑ i : Fin (n + 1), table (tokens i) coordinate) / ((n + 1 : Nat) : ℝ)

/-- Длина непустого сообщения, преобразованная в ℝ, не равна нулю. -/
theorem message_length_ne_zero (n : Nat) : ((n + 1 : Nat) : ℝ) ≠ 0 := by
  exact_mod_cast (Nat.ne_of_gt (Nat.zero_lt_succ n))

/-! ## 3. ReLU и прямой проход

ReLU выбирает большее из двух чисел: 0 и x.
Поэтому на выходе всегда значение ≥ 0, а для x ≥ 0 выход равен x.
-/

noncomputable def relu (x : ℝ) : ℝ := max 0 x

theorem relu_nonnegative (x : ℝ) : 0 ≤ relu x := by
  exact le_max_left 0 x

theorem relu_of_nonnegative (x : ℝ) (hx : 0 ≤ x) : relu x = x := by
  exact max_eq_right hx

/-- Весовые коэффициенты двухслойной сети. Размерности определены типами. -/
structure Network (inputSize hiddenSize classCount : Nat) where
  w1 : Mat hiddenSize inputSize
  b1 : Vec hiddenSize
  w2 : Mat classCount hiddenSize
  b2 : Vec classCount

/-- Скрытый слой: h = ReLU(W₁x + b₁). -/
noncomputable def hiddenLayer {inputSize hiddenSize classCount : Nat}
    (net : Network inputSize hiddenSize classCount) (x : Vec inputSize) :
    Vec hiddenSize :=
  fun i => relu (addVec (matVec net.w1 x) net.b1 i)

/-- Выход до softmax: z = W₂h + b₂. -/
noncomputable def logits {inputSize hiddenSize classCount : Nat}
    (net : Network inputSize hiddenSize classCount) (x : Vec inputSize) :
    Vec classCount :=
  addVec (matVec net.w2 (hiddenLayer net x)) net.b2

/-- Каждый элемент скрытого слоя неотрицателен при любых весах и входе. -/
theorem hidden_nonnegative {inputSize hiddenSize classCount : Nat}
    (net : Network inputSize hiddenSize classCount) (x : Vec inputSize)
    (i : Fin hiddenSize) : 0 ≤ hiddenLayer net x i := by
  exact relu_nonnegative _

/-! ## 4. Softmax над ℝ

Используем `Fin (n + 1)`, чтобы категорий было хотя бы одна. Экспонента
любого вещественного числа положительна, поэтому знаменатель больше нуля.
Это утверждение о математической модели, а не о float32 и не о качестве ИИ.
-/

noncomputable def expSum {n : Nat} (z : Vec (n + 1)) : ℝ :=
  ∑ j : Fin (n + 1), Real.exp (z j)

noncomputable def softmax {n : Nat} (z : Vec (n + 1)) : Vec (n + 1) :=
  fun i => Real.exp (z i) / expSum z

/-- Положительность знаменателя для любого набора вещественных логитов. -/
theorem expSum_pos {n : Nat} (z : Vec (n + 1)) : 0 < expSum z := by
  unfold expSum
  apply Finset.sum_pos
  · intro i _
    exact Real.exp_pos (z i)
  · exact ⟨⟨0, Nat.zero_lt_succ n⟩, Finset.mem_univ _⟩

/-- Каждый выход softmax строго положителен. -/
theorem softmax_pos {n : Nat} (z : Vec (n + 1)) (i : Fin (n + 1)) :
    0 < softmax z i := by
  exact div_pos (Real.exp_pos (z i)) (expSum_pos z)

/-- Сумма оценок softmax в точной математике равна 1. -/
theorem softmax_sum_one {n : Nat} (z : Vec (n + 1)) :
    (∑ i : Fin (n + 1), softmax z i) = 1 := by
  unfold softmax
  rw [← Finset.sum_div]
  change expSum z / expSum z = 1
  exact div_self (ne_of_gt (expSum_pos z))

/-! ## 5. Порог и отказ

Для простоты функция получает УЖЕ ВЫБРАННУЮ категорию и её оценку.
Она не вычисляет argmax: правильность выбора максимума здесь не доказана.
Порог определён для всех ℝ; на практике настройку следует ограничить [0, 1].
-/

noncomputable def decideCategory {K : Nat}
    (candidate : Fin K) (score threshold : ℝ) : Option (Fin K) :=
  if score ≥ threshold then some candidate else none

/-- Если оценка меньше порога, функция обязана вернуть отказ. -/
theorem reject_below_threshold {K : Nat} (candidate : Fin K)
    (score threshold : ℝ) (h : score < threshold) :
    decideCategory candidate score threshold = none := by
  simp [decideCategory, not_le_of_gt h]

/-- Если получена категория, её оценка не меньше порога. -/
theorem accepted_above_threshold {K : Nat} (candidate selected : Fin K)
    (score threshold : ℝ)
    (h : decideCategory candidate score threshold = some selected) :
    threshold ≤ score := by
  by_contra hNot
  have hLow : score < threshold := lt_of_not_ge hNot
  have hNone := reject_below_threshold candidate score threshold hLow
  rw [hNone] at h
  cases h

/-! ## 6. Абстрактная модель подтверждения изменения

Proposal — описание предлагаемого действия. Здесь Nat — условные
идентификаторы, а не реальные пути или криптографические хеши.
Approval привязано к полному Proposal. Поле currentVersion — версия
файла, увиденная непосредственно перед разрешением изменения.
Реальный контроль путей, файлов и гонок здесь НЕ моделируется.
-/

structure Proposal where
  actionId : Nat
  fileId : Nat
  proposedVersion : Nat
  deriving DecidableEq, Repr

structure Approval where
  proposal : Proposal
  deriving DecidableEq, Repr

/-- Все условия заданы как параметры; проверка этих фактов вне модели. -/
def mayApply (p : Proposal) (approval : Approval) (currentVersion : Nat)
    (supported validated : Bool) : Bool :=
  supported && validated && (approval.proposal == p) &&
    (currentVersion == p.proposedVersion)

/-- Если действие разрешено, флаг поддержки был установлен. -/
theorem apply_requires_supported (p : Proposal) (a : Approval) (v : Nat)
    (supported validated : Bool)
    (h : mayApply p a v supported validated = true) : supported = true := by
  simp [mayApply] at h
  aesop

/-- Если действие разрешено, флаг проверки был установлен. -/
theorem apply_requires_validated (p : Proposal) (a : Approval) (v : Nat)
    (supported validated : Bool)
    (h : mayApply p a v supported validated = true) : validated = true := by
  simp [mayApply] at h
  aesop

/-- При отсутствии поддержки действие отклоняется. -/
theorem unsupported_denied (p : Proposal) (a : Approval) (v : Nat)
    (validated : Bool) : mayApply p a v false validated = false := by
  simp [mayApply]

/-- Без успешной проверки действие отклоняется. -/
theorem unvalidated_denied (p : Proposal) (a : Approval) (v : Nat)
    (supported : Bool) : mayApply p a v supported false = false := by
  simp [mayApply]

/-- Подтверждение другого предложения не разрешает данное действие. -/
theorem different_approval_denied (p : Proposal) (a : Approval) (v : Nat)
    (supported validated : Bool) (h : a.proposal ≠ p) :
    mayApply p a v supported validated = false := by
  simp [mayApply, h]

/-- Изменившаяся версия файла делает прежнее подтверждение недействительным. -/
theorem changed_version_denied (p : Proposal) (a : Approval) (v : Nat)
    (supported validated : Bool) (h : v ≠ p.proposedVersion) :
    mayApply p a v supported validated = false := by
  simp [mayApply, h]

/-! Важное ограничение: mayApply — лишь булево решение. Эта модель не
содержит фактической записи, не устраняет гонки доступа к файлу и не доказывает,
что supported/validated выставлены честно. Эти свойства надо специфицировать
и проверять отдельно при проектировании и реализации PGR Core.
-/

end PGR
