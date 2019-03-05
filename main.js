var app = new Vue({
	el: '#app',
	data: {
		product: 'Socks',
		image: 'blue.png',
		inStock: false,
		details: ['Warm', 'Fuzzy', 'Okay not that Fuzzy', 'Still Cool'],
		cart: 0
	},
	methods: {
		addToCart: function () {
			this.cart += 1
		}
	}
})