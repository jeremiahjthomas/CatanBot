// interface StandardDiceDeck {
//     totalDice: number
//     dicePairs: DicePair[]
// }
 
// interface WeightedDiceDeck {
//     totalDice: number
//     dicePairs: DicePair[]
//     probabilityWeighting: number
//     recentlyRolledCount: number
// }
 
// export class DiceControllerBalanced extends BaseDiceController {
//     private readonly minimumCardsBeforeReshuffling: number
//     private readonly probabilityReductionForRecentlyRolled: number
//     private readonly probabilityReductionForSevenStreaks: number
 
//     private weightedDiceDeck: WeightedDiceDeck[]
//     private cardsLeftInDeck: number
//     private readonly recentRolls: number[]
//     private readonly maximumRecentRollMemory: number
//     private readonly weightedEventDiceDeck: DiceControllerBalancedCKEventDie
 
//     private readonly sevenStreakCount: {playerColor: PlayerColor, streakCount: number}
//     private readonly totalSevensRolledByPlayer: Map<PlayerColor, number>
 
//     private readonly logger: Logger
//     private readonly numberOfPlayers: number
 
//     constructor(gameController: GameController) {
//         super()
 
//         this.logger = gameController.logger
//         this.numberOfPlayers = gameController.players.length
 
//         this.initWeightedDiceDeck()
//         this.reshuffleWeightedDiceDeck()
//         this.updateWeightedDiceDeckProbabilities()
 
//         this.minimumCardsBeforeReshuffling = 13
//         this.probabilityReductionForRecentlyRolled = 0.34
//         this.probabilityReductionForSevenStreaks = 0.4
 
//         this.recentRolls = []
//         this.maximumRecentRollMemory = 5
 
//         this.sevenStreakCount = {playerColor: PlayerColor.None, streakCount: 0}
//         this.totalSevensRolledByPlayer = new Map<PlayerColor, number>()
 
//         this.weightedEventDiceDeck = new DiceControllerBalancedCKEventDie(this.logger)
//     }
 
//     throwDice(playerColor: PlayerColor): DicePair {
//         this.initTotalSevens(playerColor)
//         return this.drawWeightedCard(playerColor)
//     }
 
//     throwEventDice(): number {
//         return this.weightedEventDiceDeck.throwEventDice()
//     }
 
//     private drawWeightedCard(playerColor: PlayerColor): DicePair {
//         if(this.cardsLeftInDeck < this.minimumCardsBeforeReshuffling) this.reshuffleWeightedDiceDeck()
//         this.updateWeightedDiceDeckProbabilities()
//         this.adjustWeightedDiceDeckBasedOnRecentRolls()
//         this.adjustSevenProbabilityBasedOnSevens(playerColor)
//         return this.getWeightedDice(playerColor)
//     }
 
//     private initWeightedDiceDeck() {
//         this.weightedDiceDeck = []
//         this.weightedDiceDeck.push({totalDice: 2, dicePairs: [], probabilityWeighting: 0, recentlyRolledCount: 0})
//         this.weightedDiceDeck.push({totalDice: 3, dicePairs: [], probabilityWeighting: 0, recentlyRolledCount: 0})
//         this.weightedDiceDeck.push({totalDice: 4, dicePairs: [], probabilityWeighting: 0, recentlyRolledCount: 0})
//         this.weightedDiceDeck.push({totalDice: 5, dicePairs: [], probabilityWeighting: 0, recentlyRolledCount: 0})
//         this.weightedDiceDeck.push({totalDice: 6, dicePairs: [], probabilityWeighting: 0, recentlyRolledCount: 0})
//         this.weightedDiceDeck.push({totalDice: 7, dicePairs: [], probabilityWeighting: 0, recentlyRolledCount: 0})
//         this.weightedDiceDeck.push({totalDice: 8, dicePairs: [], probabilityWeighting: 0, recentlyRolledCount: 0})
//         this.weightedDiceDeck.push({totalDice: 9, dicePairs: [], probabilityWeighting: 0, recentlyRolledCount: 0})
//         this.weightedDiceDeck.push({totalDice: 10, dicePairs: [], probabilityWeighting: 0, recentlyRolledCount: 0})
//         this.weightedDiceDeck.push({totalDice: 11, dicePairs: [], probabilityWeighting: 0, recentlyRolledCount: 0})
//         this.weightedDiceDeck.push({totalDice: 12, dicePairs: [], probabilityWeighting: 0, recentlyRolledCount: 0})
//     }
 
//     private reshuffleWeightedDiceDeck() {
//         const standardDiceDeck = DiceControllerBalanced.getStandardDiceDeck()
 
//         for(const [totalDiceIndex, dicePairsForTotalDice] of standardDiceDeck.entries()) {
//             this.weightedDiceDeck[totalDiceIndex].dicePairs = dicePairsForTotalDice.dicePairs
//         }
 
//         const totalCombinations = 36
//         this.cardsLeftInDeck = totalCombinations
//     }
 
//     private updateWeightedDiceDeckProbabilities() {
//         for(const diceDeckForTotalDice of this.weightedDiceDeck) {
//             diceDeckForTotalDice.probabilityWeighting = diceDeckForTotalDice.dicePairs.length / this.cardsLeftInDeck
//         }
//     }
 
//     private getWeightedDice(playerColor: PlayerColor): DicePair {
//         const totalProbabilityWeight = this.getTotalProbabilityWeight()
 
//         let targetRandomNumber = Math.random() * totalProbabilityWeight
//         for(const diceDeckForTotalDice of this.weightedDiceDeck) {
//             if(targetRandomNumber <= diceDeckForTotalDice.probabilityWeighting) {
//                 const drawnCard = ArrayUtils.randomElementFromArray(diceDeckForTotalDice.dicePairs)
//                 ArrayUtils.removeElementFromArray(diceDeckForTotalDice.dicePairs, drawnCard)
 
//                 this.recentRolls.push(diceDeckForTotalDice.totalDice)
//                 diceDeckForTotalDice.recentlyRolledCount += 1
//                 this.cardsLeftInDeck -= 1
 
//                 if(this.recentRolls.length > this.maximumRecentRollMemory) this.updateRecentlyRolled()
//                 if(diceDeckForTotalDice.totalDice == 7) this.updateSevenRolls(playerColor)
//                 return drawnCard
//             }
//             targetRandomNumber -= diceDeckForTotalDice.probabilityWeighting
//         }
 
//         this.logger.logError('Something seriously wrong with weighted dice deck')
//         const defaultRollIfError = {dice1: 3, dice2: 4}
//         return defaultRollIfError
//     }
 
//     private getTotalProbabilityWeight(): number {
//         let totalProbabilityWeight = 0
//         for(const dicePairs of this.weightedDiceDeck) {
//             totalProbabilityWeight += dicePairs.probabilityWeighting
//         }
 
//         return totalProbabilityWeight
//     }
 
//     private updateRecentlyRolled() {
//         const ignore0and1 = 2
//         const totalDiceFiveRollsAgo = this.recentRolls[0]
//         this.weightedDiceDeck[totalDiceFiveRollsAgo - ignore0and1].recentlyRolledCount -= 1
//         this.recentRolls.shift()
//     }
 
//     private adjustWeightedDiceDeckBasedOnRecentRolls() {
//         for(const diceDeckForTotalDice of this.weightedDiceDeck) {
//             const probabilityReduction = (diceDeckForTotalDice.recentlyRolledCount * this.probabilityReductionForRecentlyRolled)
//             const probabilityMultiplier = 1 - probabilityReduction
//             diceDeckForTotalDice.probabilityWeighting *= probabilityMultiplier
//             if(diceDeckForTotalDice.probabilityWeighting < 0) diceDeckForTotalDice.probabilityWeighting = 0
//         }
//     }
 
//     private initTotalSevens(playerColor: PlayerColor) {
//         if(this.totalSevensRolledByPlayer.get(playerColor) != undefined) return
//         this.totalSevensRolledByPlayer.set(playerColor, 0)
//     }
 
//     private updateSevenRolls(playerColor: PlayerColor) {
//         const sevensRolledByPlayer = this.totalSevensRolledByPlayer.get(playerColor) ?? 0
//         this.totalSevensRolledByPlayer.set(playerColor, sevensRolledByPlayer + 1)
 
//         if(playerColor == this.sevenStreakCount.playerColor) {
//             this.sevenStreakCount.streakCount += 1
//             return
//         }
 
//         this.sevenStreakCount.playerColor = playerColor
//         this.sevenStreakCount.streakCount = 1
//     }
 
//     protected adjustSevenProbabilityBasedOnSevens(playerColor: PlayerColor) {
//         if(this.numberOfPlayers < 2) return
//         const streakAdjustmentPercentage = this.getStreakAdjustmentConstant(playerColor)
//         const playerSevensAdjustmentPercentage = this.getSevenImbalanceAdjustment(playerColor)
 
//         let sevenProbabilityAdjustment = 1 * playerSevensAdjustmentPercentage + streakAdjustmentPercentage
 
//         const minimumAdjustment = 0
//         const maximumAdjustment = 2
//         if(sevenProbabilityAdjustment < minimumAdjustment) sevenProbabilityAdjustment = minimumAdjustment
//         if(sevenProbabilityAdjustment > maximumAdjustment) sevenProbabilityAdjustment = maximumAdjustment
 
//         const ignore0and1 = 2
//         const sevenIndex = 7 - ignore0and1
//         this.weightedDiceDeck[sevenIndex].probabilityWeighting *= sevenProbabilityAdjustment
//     }
 
//     private getStreakAdjustmentConstant(player: PlayerColor): number {
//         const isStreakForOrAgainstPlayer = this.sevenStreakCount.playerColor == player ? -1 : 1
//         return this.probabilityReductionForSevenStreaks * this.sevenStreakCount.streakCount * isStreakForOrAgainstPlayer
//     }
 
//     private getSevenImbalanceAdjustment(playerColor: PlayerColor): number {
//         const totalSevens = this.getTotalSevensRolled()
//         if(totalSevens < this.totalSevensRolledByPlayer.size) return 1
 
//         const sevensPerPlayer = this.totalSevensRolledByPlayer.get(playerColor) ?? 0
 
//         const percentageOfTotalSevens = sevensPerPlayer / totalSevens
//         const idealPercentageOfTotalSevens = 1 / this.totalSevensRolledByPlayer.size
 
//         const adjustmentBecauseOfSevensImbalance = 1 + ((idealPercentageOfTotalSevens - percentageOfTotalSevens) / idealPercentageOfTotalSevens)
 
//         return adjustmentBecauseOfSevensImbalance
//     }
 
//     private getTotalSevensRolled(): number {
//         let totalSevensRolled = 0
//         for(const totalSevensRolledByPlayer of this.totalSevensRolledByPlayer.values()) {
//             totalSevensRolled += totalSevensRolledByPlayer
//         }
 
//         return totalSevensRolled
//     }
 
//     private static getStandardDiceDeck(): StandardDiceDeck[] {
//         const standardDiceDeck: StandardDiceDeck[] = []
//         standardDiceDeck.push({totalDice: 2, dicePairs: [{dice1: 1, dice2: 1}]})
//         standardDiceDeck.push({totalDice: 3, dicePairs: [{dice1: 1, dice2: 2}, {dice1: 2, dice2: 1}]})
//         standardDiceDeck.push({totalDice: 4, dicePairs: [{dice1: 1, dice2: 3}, {dice1: 2, dice2: 2}, {dice1: 3, dice2: 1}]})
//         standardDiceDeck.push({totalDice: 5, dicePairs: [{dice1: 1, dice2: 4}, {dice1: 2, dice2: 3}, {dice1: 3, dice2: 2}, {dice1: 4, dice2: 1}]})
//         standardDiceDeck.push({totalDice: 6, dicePairs: [{dice1: 1, dice2: 5}, {dice1: 2, dice2: 4}, {dice1: 3, dice2: 3}, {dice1: 4, dice2: 2}, {dice1: 5, dice2: 1}]})
//         standardDiceDeck.push({totalDice: 7, dicePairs: [{dice1: 1, dice2: 6}, {dice1: 2, dice2: 5}, {dice1: 3, dice2: 4}, {dice1: 4, dice2: 3}, {dice1: 5, dice2: 2}, {dice1: 6, dice2: 1}]})
//         standardDiceDeck.push({totalDice: 8, dicePairs: [{dice1: 2, dice2: 6}, {dice1: 3, dice2: 5}, {dice1: 4, dice2: 4}, {dice1: 5, dice2: 3}, {dice1: 6, dice2: 2}]})
//         standardDiceDeck.push({totalDice: 9, dicePairs: [{dice1: 3, dice2: 6}, {dice1: 4, dice2: 5}, {dice1: 5, dice2: 4}, {dice1: 6, dice2: 3}]})
//         standardDiceDeck.push({totalDice: 10, dicePairs: [{dice1: 4, dice2: 6}, {dice1: 5, dice2: 5}, {dice1: 6, dice2: 4}]})
//         standardDiceDeck.push({totalDice: 11, dicePairs: [{dice1: 5, dice2: 6}, {dice1: 6, dice2: 5}]})
//         standardDiceDeck.push({totalDice: 12, dicePairs: [{dice1: 6, dice2: 6}]})
 
//         return standardDiceDeck
//     }
// }