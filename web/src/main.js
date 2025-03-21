let generateCartSummary = () => {
  cartItems.innerHTML = basket.map(item => {
    let product = shopItemsData.find(p => p.id === item.id);
    return` 
    <div class="cart-item">
  <img width="100" src="${product.img}" alt="${product.name}" class="cart-item-img">
  <div class="details">
    <div class="title-price-x">
      <h4 class="product-name">${product.name}</h4>
    </div>

    <div class="total-price">
      <strong>Price:</strong> £${(product.price).toFixed(2)}
    </div>

    <div class="quantity-info">
      <strong>Quantity:</strong> ${item.item}
    </div>    
  </div>
</div>
`;
  }).join("") || `<h2>Cart is Empty</h2>`;
};

function initializeSocket() {
  try {
    window.socket = io("http://localhost:5001", {
      transports: ["websocket"],
    });

    console.log("✅ Socket.IO connected (1st).");

    window.socket.on("connect", () => {
      console.log("✅ Socket connected verified");
      console.log("socket.connected after connect event:", window.socket.connected);
    })

    window.socket.on("cart_update", (data) => {
      console.log("Received cart update:", data);
      const { productId, quantity } = data;

      let search = basket.find(x => x.id === productId);

      if (search) {
        search.item += quantity;
      } else {
        basket.push({ id: productId, item: quantity });
      }

      updateCart();
    });

    window.socket.on("connect_error", (error) => {
      console.error("❌ Socket.IO connection error:", error);
    });

    window.socket.on("disconnect", () => {
      console.warn("⚠️ Socket.IO disconnected.");
    });

  } catch (error) {
    console.error("❌ Failed to initialize Socket.IO:", error);
  }
}

function getQuantityById(id) {
  const cartData = localStorage.getItem("data");

  if (!cartData) {
    console.warn("Cart is empty.");
    return 0;
  }

  try {
    const parsedCart = JSON.parse(cartData);

    if (Array.isArray(parsedCart)) {
      const product = parsedCart.find(item => item.id === id);
      if (product) {
        return product.item;
      } else {
        console.warn(`Item with id=${id} not found in cart.`);
        return 0;
      }
    } else {
      console.error("Cart data is not an array.");
      return 0;
    }
  } catch (error) {
    console.error("Error parsing cart data:", error);
    return 0;
  }
}

function handleCheckout() {
  const totalAmount = basket.reduce((sum, item) => {
    const product = shopItemsData.find(p => p.id === item.id);
    return sum + (item.item * product.price);
  }, 0);

  if (totalAmount > 0) {
    window.location.href = "payment.html";
  } else {
    alert("Your cart is empty! Please add items before checking out.");
  }
}

const sendPurchasedQuantity = async () => {
  const cartData = localStorage.getItem("data");

  let cartQuantity = 0;

  if (cartData) {
    try {
      const parsedCart = JSON.parse(cartData);
      if (Array.isArray(parsedCart)) {
        const product = parsedCart.find(item => item.id === 1);
        if (product) {
          cartQuantity = product.item;
        }
      }
    } catch (error) {
      console.error("Error parsing cart data:", error);
    }
  }

  console.log("✅ Sending Purchased Quantity for id=1:", cartQuantity);

  try {
    const response = await fetch(`http://localhost:5001/confirm_purchase?quantity=${cartQuantity}`, {
      method: "GET",
    });
    const result = await response.text();
    console.log("Server response:", result);
  } catch (error) {
    console.error("Error sending purchase data:", error);
  }
};

const shop = document.getElementById("shop");
const cartItems = document.getElementById("cart-items");
let basket = JSON.parse(localStorage.getItem("data")) || [];

window.increment = function(id) {
  let search = basket.find(x => x.id === id);
  if (search) {
    search.item += 1;
  } else {
    basket.push({ id: id, item: 1 });
  }
  updateCart();
};

window.decrement = function(id) {
  let search = basket.find(x => x.id === id);
  if (!search) return;
  search.item -= 1;
  basket = basket.filter(x => x.item > 0);
  updateCart();
};

window.removeItem = function(id) {
  basket = basket.filter(x => x.id !== id);
  updateCart();
};

window.clearCart = function() {
  basket = [];
  localStorage.setItem("data", "[]");
  updateCart();

  if (typeof BroadcastChannel !== "undefined") {
    const channel = new BroadcastChannel("cart_channel");
    channel.postMessage({ type: "cart_cleared" });
  }
};

const generateShop = () => {
  if (!shop) return;
  
  shop.innerHTML = shopItemsData.map(item => {
    const { id, name, price, desc, img } = item;
    const search = basket.find(x => x.id === id) || { item: 0 };
    
    return `
    <div id="product-${id}" class="item">
      <img width="220" src="${img}" alt="${name}" onclick="increment(${id})">
      <div class="details">
        <h3>${name}</h3>
        <p>${desc}</p>
        <div class="price-quantity">
          <h2>£ ${price.toFixed(2)}</h2>
          <div class="buttons">
            <i onclick="decrement(${id})" class="bi bi-dash-lg"></i>
            <div id="${id}" class="quantity">${search.item}</div>
            <i onclick="increment(${id})" class="bi bi-plus-lg"></i>
          </div>
        </div>
      </div>
    </div>`;
  }).join("");
};

const updateCart = () => {
  generateShop();
  generateCartItems();
  updateTotal();
  localStorage.setItem("data", JSON.stringify(basket));
};

const generateCartItems = () => {
  if (!cartItems) return;
  
  cartItems.innerHTML = basket.length ? basket.map(item => {
    const product = shopItemsData.find(p => p.id === item.id);
    return ` 
    <div class="cart-item">
      <img width="100" src="${product.img}" alt="${product.name}">
      <div class="details">
        <div class="title-price-x">
            <h4>${product.name}</h4>
        </div>
        <h4 class="cart-item-price">£ ${product.price}</h4>
        <div class="buttons">
          
          <i onclick="decrement(${item.id})" class="bi bi-dash-lg"></i>
          <div class="quantity">${item.item}</div>
          <i onclick="increment(${item.id})" class="bi bi-plus-lg"></i>
          <i onclick="removeItem(${item.id})" class="bi bi-trash"></i>
        </div>
        <h4>£ ${(item.item * product.price).toFixed(2)}</h4>
      </div>
    </div>`;
  }).join("") : `<h2>Cart is Empty</h2>`;
};

const updateTotal = () => {
  const total = basket.reduce((sum, item) => {
    const product = shopItemsData.find(p => p.id === item.id);
    return sum + (item.item * product.price);
  }, 0);

  document.querySelectorAll("#total-amount").forEach(element => {
    element.textContent = total.toFixed(2);
  });

  document.querySelectorAll("#cartAmount").forEach(element => {
    element.textContent = basket.reduce((a, b) => a + b.item, 0);
  });
};

document.addEventListener("DOMContentLoaded", () => {
  console.log("Page loaded, initializing shop...");

  generateShop();
  generateCartItems();
  updateTotal();

  initializeSocket();
});
